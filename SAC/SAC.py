import random
import csv
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pandas as pd
import os


# -------------------- Replay Buffer -------------------- #
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


# -------------------- Networks -------------------- #
class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.mu_layer = init_layer_uniform(nn.Linear(128, out_dim))
        self.log_std_layer = init_layer_uniform(nn.Linear(128, out_dim))

    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        mu = self.mu_layer(x).tanh()
        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class CriticQ(nn.Module):
    def __init__(self, in_dim):
        super(CriticQ, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = init_layer_uniform(nn.Linear(128, 1))

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)


class CriticV(nn.Module):
    def __init__(self, in_dim):
        super(CriticV, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = init_layer_uniform(nn.Linear(128, 1))

    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        return self.out(x)


# -------------------- SAC Agent -------------------- #
class SACAgent:
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = int(1e4),
        policy_update_freq: int = 2,
        seed: int = 777,
    ):
        self.env = env
        self.gamma, self.tau = gamma, tau
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.memory = ReplayBuffer(obs_dim, action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.target_entropy = -np.prod((action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_target = CriticV(obs_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())
        self.qf_1 = CriticQ(obs_dim + action_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=3e-4)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=3e-4)

        self.transition, self.total_step, self.is_test = list(), 0, False

    def select_action(self, state: np.ndarray):
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                self.actor(torch.FloatTensor(state).to(self.device))[0].detach().cpu().numpy()
            )
        self.transition = [state, selected_action]
        return selected_action

    def step(self, action: np.ndarray):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.FloatTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        new_action, log_prob = self.actor(state)

        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        mask = 1 - done
        q_1_pred = self.qf_1(state, action)
        q_2_pred = self.qf_2(state, action)
        v_target = self.vf_target(next_state)
        q_target = reward + self.gamma * v_target * mask
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        v_pred = self.vf(state)
        q_pred = torch.min(self.qf_1(state, new_action), self.qf_2(state, new_action))
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % self.policy_update_freq == 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(())

        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()
        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss
        return actor_loss.item(), qf_loss.item(), vf_loss.item(), alpha_loss.item()

    def train(self, num_frames: int, log_file="sac_log.csv", log_interval=1000):
        self.is_test = False
        state, _ = self.env.reset(seed=self.seed)
        scores, score = [], 0

        # init csv
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "episode", "reward", "actor_loss", "qf_loss", "vf_loss", "alpha_loss"])

        episode = 0
        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward

            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                episode += 1
                score = 0

            if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                actor_loss, qf_loss, vf_loss, alpha_loss = self.update_model()
                # log only at end of each episode
                if done:
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([self.total_step, episode, scores[-1], actor_loss, qf_loss, vf_loss, alpha_loss])

            if episode > 0 and episode % 50 == 0 and done:
                avg_reward = np.mean(scores[-50:])
                print(f"[Episode {episode}] AvgReward(last50)={avg_reward:.2f}")

        self.env.close()

        # visualize
        df = pd.read_csv(log_file)
        plt.figure(figsize=(15, 6))
        plt.subplot(231); plt.title("Reward"); plt.plot(df["episode"], df["reward"])
        plt.subplot(232); plt.title("Actor Loss"); plt.plot(df["episode"], df["actor_loss"])
        plt.subplot(233); plt.title("Qf Loss"); plt.plot(df["episode"], df["qf_loss"])
        plt.subplot(234); plt.title("Vf Loss"); plt.plot(df["episode"], df["vf_loss"])
        plt.subplot(235); plt.title("Alpha Loss"); plt.plot(df["episode"], df["alpha_loss"])
        plt.tight_layout(); plt.show()

    def test(self, video_folder="videos/sac"):
        self.is_test = True
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        state, _ = env.reset(seed=self.seed)
        done, score = False, 0
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
        env.close()
        print("Test score:", score)
        print(f"Video saved to {video_folder}")

    def _target_soft_update(self):
        tau = self.tau
        for t_param, l_param in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


# -------------------- Action Normalizer -------------------- #
class ActionNormalizer(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        low, high = self.action_space.low, self.action_space.high
        scale, reloc = (high - low) / 2, high - (high - low) / 2
        return np.clip(action * scale + reloc, low, high)

    def reverse_action(self, action: np.ndarray):
        low, high = self.action_space.low, self.action_space.high
        scale, reloc = (high - low) / 2, high - (high - low) / 2
        return np.clip((action - reloc) / scale, -1.0, 1.0)


# -------------------- Main -------------------- #
if __name__ == "__main__":
    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("Pendulum-v1", render_mode=None)
    env = ActionNormalizer(env)

    agent = SACAgent(env, memory_size=100000, batch_size=128, initial_random_steps=10000, seed=seed)

    # Train
    agent.train(num_frames=50000)

    # Test
    agent.test("videos/sac")
