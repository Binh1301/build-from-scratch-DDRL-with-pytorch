import copy
import os
import random
import csv
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

seed = 777
random.seed(seed)
np.random.seed(seed)
seed_torch(seed)

class ReplayBuffer:
    def __init__(self, state_size: int, size: int, batch_size: int = 32):
        self.state_memory = np.zeros([size, state_size], dtype=np.float32)
        self.action_memory = np.zeros([size], dtype=np.float32)
        self.reward_memory = np.zeros([size], dtype=np.float32)
        self.next_state_memory = np.zeros([size, state_size], dtype=np.float32)
        self.done_memory = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.state_memory[self.ptr] = obs
        self.action_memory[self.ptr] = act
        self.reward_memory[self.ptr] = rew
        self.next_state_memory[self.ptr] = next_obs
        self.done_memory[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.state_memory[idxs],
            next_obs=self.next_state_memory[idxs],
            acts=self.action_memory[idxs],
            rews=self.reward_memory[idxs],
            done=self.done_memory[idxs],
        )

    def __len__(self) -> int:
        return self.size

class GaussianNoise:
    def __init__(self, action_size, min_sigma=1.0, max_sigma=1.0, decay_period=1000000):
        self.action_size = action_size
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_size)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, init_w=3e-3):
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        return self.out(x).tanh()


class CriticQ(nn.Module):
    def __init__(self, state_size, init_w=3e-3):
        super(CriticQ, self).__init__()
        self.hidden1 = nn.Linear(state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

class TD3Agent:
    def __init__(self, env, memory_size, batch_size,
                 gamma=0.99, tau=5e-3,
                 exploration_noise=0.1,
                 target_policy_noise=0.2,
                 target_policy_noise_clip=0.5,
                 initial_random_steps=1e4,
                 policy_update_freq=2,
                 seed=777):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic2 = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic1_target = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic2_target = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.memory = ReplayBuffer(self.state_size, memory_size, batch_size)

        self.exploration_noise = GaussianNoise(self.action_size, exploration_noise, exploration_noise)
        self.target_policy_noise = GaussianNoise(self.action_size, target_policy_noise, target_policy_noise)
        self.target_policy_noise_clip = target_policy_noise_clip

        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(critic_params, lr=1e-3)

        self.transition = []
        self.total_step = 0
        self.is_test = False

    def get_action(self, state):
        if self.total_step < self.initial_random_steps and not self.is_test:
            action = self.env.action_space.sample()
        else:
            action = self.actor(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()

        if not self.is_test:
            noise = self.exploration_noise.sample()
            action = np.clip(action + noise, -1.0, 1.0)

        self.transition = [state, action]
        return action

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        return next_state, reward, done

    def train_step(self):
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        masks = 1 - done

        noise = torch.FloatTensor(self.target_policy_noise.sample()).to(self.device)
        noise = torch.clamp(noise, -self.target_policy_noise_clip, self.target_policy_noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q1_target = self.critic1_target(next_state, next_action)
        q2_target = self.critic2_target(next_state, next_action)
        q_target = torch.min(q1_target, q2_target)
        expected_q = reward + self.gamma * q_target * masks
        expected_q = expected_q.detach()

        critic1_loss = F.mse_loss(q1, expected_q)
        critic2_loss = F.mse_loss(q2, expected_q)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_step % self.policy_update_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.item(), critic_loss.item()

    def _target_soft_update(self):
        tau = self.tau
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)
        for t, s in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)
        for t, s in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)


class ActionNormalizer(gym.ActionWrapper):
    def action(self, action):
        low, high = self.action_space.low, self.action_space.high
        action = action * (high - low) / 2 + (high + low) / 2
        return np.clip(action, low, high)
    def reverse_action(self, action):
        low, high = self.action_space.low, self.action_space.high
        return np.clip(2 * (action - low) / (high - low) - 1, -1, 1)

env_name = "Pendulum-v1"
env = gym.make(env_name, render_mode="rgb_array")
env = ActionNormalizer(env)

num_episodes = 500
memory_size = 5000
batch_size = 32
initial_random_steps = 5000

agent = TD3Agent(env, memory_size, batch_size, initial_random_steps=initial_random_steps, seed=seed)

log_file = "td3_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "actor_loss", "critic_loss"])

scores, actor_losses, critic_losses = [], [], []

for ep in range(num_episodes):
    state, _ = agent.env.reset(seed=seed)
    ep_reward = 0
    done = False
    while not done:
        agent.total_step += 1
        action = agent.get_action(state)
        next_state, reward, done = agent.step(action)
        state = next_state
        ep_reward += reward
        if len(agent.memory) >= batch_size and agent.total_step > agent.initial_random_steps:
            a_loss, c_loss = agent.train_step()
            actor_losses.append(a_loss)
            critic_losses.append(c_loss)
    scores.append(ep_reward)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ep + 1, ep_reward,
                         np.mean(actor_losses[-10:]) if actor_losses else 0,
                         np.mean(critic_losses[-10:]) if critic_losses else 0])
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{num_episodes}, Reward: {ep_reward:.2f}")

agent.env.close()


plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Scores")
plt.plot(scores)
plt.subplot(132)
plt.title("Actor Loss")
plt.plot(actor_losses)
plt.subplot(133)
plt.title("Critic Loss")
plt.plot(critic_losses)
plt.show()
