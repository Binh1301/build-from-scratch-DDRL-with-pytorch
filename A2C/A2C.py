import random
import csv
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pandas as pd


def seed_torch(seed): # Hàm để thiết lập seed cho torch
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


seed = 777
random.seed(seed)
np.random.seed(seed)
seed_torch(seed)


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3): # Khởi tạo trọng số đều
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module): # Mạng chính sách
    def __init__(self, state_size: int, action_size: int):
        super(Actor, self).__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.mu_layer = nn.Linear(128, action_size)
        self.log_std_layer = nn.Linear(128, action_size)
        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.hidden(state))
        mu = torch.tanh(self.mu_layer(x)) * 2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        return action, dist


class CriticV(nn.Module):   # Hàm xấp xỉ hàm giá trị V value
    def __init__(self, state_size: int):
        super(CriticV, self).__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.out = nn.Linear(128, 1)
        initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor):
        value = F.relu(self.hidden(state))
        value = self.out(value)
        return value


class A2CAgent: # Đại lý A2C
    def __init__(self, env: gym.Env, gamma: float, entropy_weight: float, seed: int = 777):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = CriticV(self.state_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.transition: list = list()
        self.total_step = 0
        self.is_test = False

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if not self.is_test:
            self.transition.extend([next_state, reward, done])
        return next_state, reward, done

    def train_step(self):
        state, log_prob, next_state, reward, done = self.transition

        mask = 1 - done
        next_state = torch.FloatTensor(next_state).to(self.device)
        curr_Q = self.critic(state)
        next_Q = self.critic(next_state)
        expected_Q = reward + self.gamma * next_Q * mask

        critic_loss = F.smooth_l1_loss(curr_Q, expected_Q.detach())
        advantage = (expected_Q - curr_Q).detach()
        actor_loss = -advantage * log_prob
        actor_loss += self.entropy_weight * -log_prob

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()




# Environment
env = gym.make("Pendulum-v1", render_mode=None)

# Parameters
num_frames = 100000
gamma = 0.9
entropy_weight = 1e-2
max_episodes = 500
log_file = "a2c_log.csv"

# Init agent
agent = A2CAgent(env, gamma, entropy_weight, seed)

# Create log file with header
with open(log_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "frame_idx", "reward", "actor_loss", "critic_loss"])

if __name__ == "__main__":
    agent.is_test = False
    actor_losses, critic_losses, scores = [], [], []
    frame_idx = 0

    for episode in range(1, max_episodes + 1):
        state, _ = agent.env.reset(seed=agent.seed)
        episode_reward = 0
        done = False

        while not done:
            frame_idx += 1
            action = agent.get_action(state)
            next_state, reward, done = agent.step(action)

            state = next_state
            episode_reward += reward

            actor_loss, critic_loss = agent.train_step()

        # Save to CSV
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, frame_idx, episode_reward, actor_loss, critic_loss])

        scores.append(episode_reward)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        # Print log mỗi 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(scores[-50:])
            print(f"[Episode {episode}/{max_episodes}] "
                  f"AvgReward(last50)={avg_reward:.2f}, "
                  f"ActorLoss={actor_loss:.4f}, CriticLoss={critic_loss:.4f}")

    agent.env.close()

    # ---------------- VISUALIZE ---------------- #
    df = pd.read_csv(log_file)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title("Rewards")
    plt.plot(df["episode"], df["reward"])

    plt.subplot(132)
    plt.title("Actor Loss")
    plt.plot(df["episode"], df["actor_loss"])

    plt.subplot(133)
    plt.title("Critic Loss")
    plt.plot(df["episode"], df["critic_loss"])

    plt.tight_layout()
    plt.show()
