import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt
import os


class PolicyNet(nn.Module): # Mạng chính sách
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        # clamp để tránh nan
        return torch.softmax(logits, dim=-1).clamp(min=1e-8, max=1-1e-8)


class ValueNet(nn.Module): # Hàm xấp xỉ hàm giá trị
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

def conjugate_gradient(Avp, b, nsteps=10, residual_tol=1e-10): # Giải hệ phương trình Ax = b
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    for _ in range(nsteps):
        Ap = Avp(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if rsnew < residual_tol:
            break
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
    return x

def flat_params(model): # Trả về tham số của mô hình dưới dạng vector phẳng
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, new_params):
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(new_params[idx:idx + size].view_as(p))
        idx += size

def flat_grad(y, model, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(
        y, model.parameters(),
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True
    )
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, model.parameters())]
    return torch.cat([g.contiguous().view(-1) for g in grads])

def train_trpo(env_name="CartPole-v1", episodes=200, gamma=0.99, max_kl=5e-3, cg_iters=10,
               backtrack_coeff=0.8, backtrack_iters=10, damping=0.1, log_file="trpo_log.csv"):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim)
    value_fn = ValueNet(obs_dim)
    value_optimizer = optim.Adam(value_fn.parameters(), lr=1e-3)

    # CSV log file
    if os.path.exists(log_file):
        os.remove(log_file)
    f = open(log_file, mode="w", newline="")
    writer = csv.writer(f)
    writer.writerow(["Episode", "Reward", "Value_Loss", "Policy_Loss", "KL", "Accepted"])

    rewards_history = []

    for ep in range(episodes): # Mỗi episode
        obs, _ = env.reset()
        obs_list, act_list, rew_list = [], [], []
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            probs = policy(obs_t)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)
            obs = next_obs

        # Compute returns
        returns, G = [], 0.0
        for r in reversed(rew_list):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32)
        act_batch = torch.tensor(act_list, dtype=torch.int64)

        # Old action probabilities
        with torch.no_grad():
            old_probs = policy(obs_batch).detach()
            old_log_probs_act = torch.log(
                old_probs.gather(1, act_batch.view(-1, 1)).squeeze(1) + 1e-8
            )

        # Advantage = return - baseline
        values = value_fn(obs_batch).squeeze(1)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update value function (MSE)
        value_loss = ((values - returns) ** 2).mean()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Surrogate loss (to MINIMIZE)
        def surrogate_loss():
            new_probs = policy(obs_batch).clamp(min=1e-8, max=1-1e-8)
            new_dist = torch.distributions.Categorical(probs=new_probs)
            new_log_probs_act = new_dist.log_prob(act_batch)
            ratio = torch.exp(new_log_probs_act - old_log_probs_act)
            return -(ratio * advantages).mean()

        # True KL (old || new)
        def kl_divergence():
            new_probs = policy(obs_batch).clamp(min=1e-8, max=1-1e-8)
            kl = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))).sum(dim=1).mean()
            return kl

        # Policy gradient g
        loss_pi = surrogate_loss()
        g = flat_grad(loss_pi, policy, retain_graph=False, create_graph=False).detach()

        # Hessian-vector product
        def hessian_vector_product(v):
            kl = kl_divergence()
            grads = flat_grad(kl, policy, create_graph=True, retain_graph=True)
            kl_v = torch.dot(grads, v)
            hvp = flat_grad(kl_v, policy, retain_graph=True)
            return hvp + damping * v

        # Solve Hx = g
        step_dir = conjugate_gradient(hessian_vector_product, g, nsteps=cg_iters)
        if torch.isnan(step_dir).any():
            step_dir = -g
        shs = 0.5 * torch.dot(step_dir, hessian_vector_product(step_dir))
        if shs <= 0 or torch.isnan(shs):
            fullstep = -1e-3 * g
        else:
            lm = torch.sqrt(shs / (max_kl + 1e-8))
            fullstep = step_dir / (lm + 1e-8)

        old_params = flat_params(policy)

        # Backtracking line search
        success = False
        old_loss = loss_pi.item()
        for i in range(backtrack_iters):
            step_frac = backtrack_coeff ** i
            new_params = old_params - step_frac * fullstep
            set_params(policy, new_params)

            new_loss = surrogate_loss().item()
            kl_value = kl_divergence().item()

            if (kl_value <= max_kl) and (new_loss < old_loss):
                success = True
                break

        if not success:
            set_params(policy, old_params)  # rollback
            kl_value = 0.0
            new_loss = old_loss

        # Logging
        total_reward = float(sum(rew_list))
        rewards_history.append(total_reward)
        writer.writerow([ep + 1, total_reward, float(value_loss.item()), float(loss_pi.item()), float(kl_value), int(success)])
        print(f"Episode {ep+1:03d} | Reward: {total_reward:7.2f} | KL: {kl_value:.6f} | StepOK: {success}")

    f.close()
    env.close()

    # Visualization
    plt.figure()
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TRPO Training Reward")
    plt.tight_layout()
    plt.show()

    return policy, rewards_history

if __name__ == "__main__":
    train_trpo()
