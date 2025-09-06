import random
import csv
import os
from collections import deque
from typing import List, Tuple, Deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

"""## Set random seed"""
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

seed = 777
random.seed(seed)
np.random.seed(seed)
seed_torch(seed)


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int,
                 log_std_min: int = -20, log_std_max: int = 0):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden = nn.Linear(state_size, 128)
        self.mu_layer = initialize_uniformly(nn.Linear(128, action_size))
        self.log_std_layer = initialize_uniformly(nn.Linear(128, action_size))

    def forward(self, state: torch.Tensor):
        x = F.relu(self.hidden(state))
        mu = torch.tanh(self.mu_layer(x))
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        return action, dist


class CriticV(nn.Module):
    def __init__(self, state_size: int):
        super(CriticV, self).__init__()
        self.hidden = nn.Linear(state_size, 128)
        self.out = initialize_uniformly(nn.Linear(128, 1))

    def forward(self, state: torch.Tensor):
        value = F.relu(self.hidden(state))
        return self.out(value)


def compute_gae(next_value, rewards, masks, values, gamma, lmbda) -> List:
    values = values + [next_value]
    gae = 0
    returns: Deque[float] = deque()
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lmbda * masks[step] * gae
        returns.appendleft(gae + values[step])
    return list(returns)


def ppo_iter(epoch, mini_batch_size, states, actions, values, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], \
                  log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]


class PPOAgent:
    def __init__(self, env: gym.Env, batch_size: int, gamma: float, tau: float,
                 epsilon: float, epoch: int, rollout_len: int,
                 entropy_weight: float, seed: int = 777):

        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.entropy_weight = entropy_weight
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using:", self.device)

        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = CriticV(self.state_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []
        self.is_test = False

    def get_action(self, state: np.ndarray):
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action
        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))
        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))
        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))
        return next_state, reward, done

    def train_step(self, next_state: np.ndarray):
        device = self.device
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = self.critic(next_state)

        returns = compute_gae(next_value, self.rewards, self.masks,
                              self.values, self.gamma, self.tau)
        states = torch.cat(self.states).view(-1, self.state_size)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []
        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            self.epoch, self.batch_size, states, actions,
            values, log_probs, returns, advantages):

            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            surr_loss = ratio * adv
            clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon,
                                            1.0 + self.epsilon) * adv
            entropy = dist.entropy().mean()
            actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean() \
                         - entropy * self.entropy_weight

            curr_Q = self.critic(state)
            critic_loss = (return_ - curr_Q).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []
        return np.mean(actor_losses), np.mean(critic_losses)


class ActionNormalizer(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        low, high = self.action_space.low, self.action_space.high
        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor
        action = action * scale_factor + reloc_factor
        return np.clip(action, low, high)

    def reverse_action(self, action: np.ndarray):
        low, high = self.action_space.low, self.action_space.high
        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor
        action = (action - reloc_factor) / scale_factor
        return np.clip(action, -1.0, 1.0)


# ============ Training ============
env = gym.make("Pendulum-v1", render_mode=None)
env = ActionNormalizer(env)

agent = PPOAgent(env, gamma=0.9, tau=0.8,
                 batch_size=64, epsilon=0.2,
                 epoch=10, rollout_len=2048,
                 entropy_weight=0.005, seed=seed)

max_episodes = 300
log_file = "ppo_training_log.csv"

if os.path.exists(log_file):
    os.remove(log_file)
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Reward", "ActorLoss", "CriticLoss"])

scores, actor_losses, critic_losses = [], [], []

for episode in range(1, max_episodes + 1):
    state, _ = env.reset(seed=agent.seed)
    state = np.expand_dims(state, axis=0)
    done = False
    episode_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = agent.step(action)
        state = next_state
        episode_reward += reward[0][0]

    actor_loss, critic_loss = agent.train_step(next_state)
    scores.append(episode_reward)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, episode_reward, actor_loss, critic_loss])

    if episode % 50 == 0:
        avg50 = np.mean(scores[-50:])
        print(f"Episode {episode}/{max_episodes}, "
              f"Reward={episode_reward:.2f}, Avg50={avg50:.2f}")

env.close()

# ============ Visualization ============
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
