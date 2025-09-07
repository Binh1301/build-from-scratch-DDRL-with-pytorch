# ddpg_pytorch.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym  
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("s", "a", "r", "s2", "d"))

class ReplayBuffer:
    def __init__(self, capacity): 
        self.buffer = deque(maxlen=capacity) # tạo mảng rỗng

    def push(self, s, a, r, s2, d): # lưu
        self.buffer.append(Transition(s, a, r, s2, d))

    def sample(self, batch_size): # tách 
        batch = random.sample(self.buffer, batch_size)
        s = np.array([t.s for t in batch], dtype=np.float32)
        a = np.array([t.a for t in batch], dtype=np.float32)
        r = np.array([t.r for t in batch], dtype=np.float32).reshape(-1,1)
        s2 = np.array([t.s2 for t in batch], dtype=np.float32)
        d = np.array([t.d for t in batch], dtype=np.float32).reshape(-1,1)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

def weight_init(m): # tránh vanishing exploding
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class Actor(nn.Module): # Actor network 
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh() # chuẩn hóa đầu ra (-1,1)
        )
        self.max_action = max_action 
        self.apply(weight_init)

    def forward(self, s):
        return self.net(s) * self.max_action  # (-2,2)

class Critic(nn.Module): # Q(s,a)
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fcs = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        self.apply(weight_init)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = torch.relu(self.fcs(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device='cpu',
        gamma=0.99, # discount factor
        tau=0.995,  # polyak        
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=int(1e6), 
        batch_size=128,
        start_steps=10000, # số bước đầu chỉ random hành động để explore
        action_noise_std=0.1 # std của Gaussian noise để thêm vào action
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.action_noise_std = action_noise_std

        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor_targ = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic_targ = Critic(state_dim, action_dim).to(self.device)

        # copy weights to targets
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay
        self.replay = ReplayBuffer(buffer_size)

        # action range
        self.max_action = max_action

        # step counter
        self.total_it = 0

    def select_action(self, state, noise=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_t).cpu().data.numpy().flatten()
        if noise:
            action = action + np.random.normal(0, self.action_noise_std, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, updates=1):
        if len(self.replay) < self.batch_size:
            return

        for _ in range(updates):
            s, a, r, s2, d = self.replay.sample(self.batch_size)

            s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
            a_t = torch.tensor(a, dtype=torch.float32, device=self.device)
            r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
            s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)
            d_t = torch.tensor(d, dtype=torch.float32, device=self.device)

            # Q target
            with torch.no_grad():
                a2 = self.actor_targ(s2_t)
                q_target_next = self.critic_targ(s2_t, a2)
                y = r_t + self.gamma * (1.0 - d_t) * q_target_next # Bellman

            # MSBE
            q_val = self.critic(s_t, a_t)
            critic_loss = nn.MSELoss()(q_val, y)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            actor_loss = -self.critic(s_t, self.actor(s_t)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # polyak update
            for p, p_targ in zip(self.critic.parameters(), self.critic_targ.parameters()):
                p_targ.data.copy_(self.tau * p_targ.data + (1 - self.tau) * p.data)
            for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                p_targ.data.copy_(self.tau * p_targ.data + (1 - self.tau) * p.data)

    def store_transition(self, s, a, r, s2, d): # luu vao replay buffer
        self.replay.push(s, a, r, s2, d)
        self.total_it += 1

def train_ddpg(
    env_name="Pendulum-v1",
    seed=0,
    num_steps=200_000,
    max_episode_steps=None,
    log_interval=5000,
    **agent_kwargs
):
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps) if max_episode_steps else env
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)

    obs_space = env.observation_space
    act_space = env.action_space

    assert isinstance(act_space, gym.spaces.Box), "Action space must be continuous (Box)."

    state_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]
    max_action = float(act_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action, device='cpu', **agent_kwargs)

    total_steps = 0
    episode_returns = []
    ep_return = 0
    s, _ = env.reset()
    episode_steps = 0
    ep_num = 0

    while total_steps < num_steps:
        if total_steps < agent.start_steps:
            a = np.random.uniform(low=act_space.low, high=act_space.high, size=action_dim)
        else:
            a = agent.select_action(s, noise=True)

        s2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        agent.store_transition(s, a, r, s2, float(done))
        ep_return += r
        episode_steps += 1
        total_steps += 1

        agent.train(updates=1)

        s = s2

        if done:
            ep_num += 1
            episode_returns.append(ep_return)
            if ep_num % 10 == 0:
                avg_ret = np.mean(episode_returns[-10:])
                print(f"Step {total_steps}\tEpisode {ep_num}\tAvgReturn(last10) {avg_ret:.2f}")
            s, _ = env.reset()
            ep_return = 0
            episode_steps = 0

        if total_steps % log_interval == 0 and total_steps > 0:
            print(f"Total steps: {total_steps}, Replay size: {len(agent.replay)}")

    env.close()
    return agent

if __name__ == "__main__":
    agent = train_ddpg(
        env_name="Pendulum-v1",
        seed=42,
        num_steps=50000,
        max_episode_steps=200,
        start_steps=1000,
        action_noise_std=0.2,
        batch_size=128
    )
