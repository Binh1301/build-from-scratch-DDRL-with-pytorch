import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
# cSpell:disable
class PolicyEstimator(): ## π(a|s) hàm chính sách
    def __init__(self, env):
        self.num_observations = env.observation_space.shape[0] ## số lượng trạng thái vị trí, vận tốc, góc , tốc dộ góc
        self.num_actions = env.action_space.n ## trái hoặc phải

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions), 
            nn.Softmax(dim=-1) # xác suất của các hành động
        )

    def predict(self, observation):
        return self.network(torch.FloatTensor(observation)) # biến từ float thành tensor rồi cho chạy nn

def vanilla_policy_gradient(env, estimator, num_episodes=1500, batch_size=10, discount_factor=0.99, render=False,
                            early_exit_reward_amount=None):
    total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = optim.Adam(estimator.network.parameters(), lr=0.01)
    action_space = np.arange(env.action_space.n) # [0, 1] cho CartPole

    for current_episode in range(num_episodes):
        observation, _ = env.reset() 
        rewards, actions, observations = [], [], [] 

        while True:
            if render:
                env.render()

            action_probs = estimator.predict(observation).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)

            observations.append(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            actions.append(action)

            if done:
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                discounted_rewards = r - r.mean()

                batch_rewards.extend(discounted_rewards)
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter >= batch_size:
                    optimizer.zero_grad()

                    batch_rewards = torch.FloatTensor(batch_rewards)
                    batch_observations = torch.FloatTensor(batch_observations)
                    batch_actions = torch.LongTensor(batch_actions)

                    logprob = torch.log(estimator.predict(batch_observations))
                    batch_actions = batch_actions.reshape(len(batch_actions), 1)
                    selected_logprobs = batch_rewards * torch.gather(logprob, 1, batch_actions).squeeze()
                    loss = -selected_logprobs.mean()

                    loss.backward()
                    optimizer.step()

                    batch_rewards, batch_observations, batch_actions = [], [], []
                    batch_counter = 1

                average_reward = np.mean(total_rewards[-100:])
                if current_episode % 100 == 0:
                    print(f"average of last 100 rewards as of episode {current_episode}: {average_reward:.2f}")

                if early_exit_reward_amount and average_reward > early_exit_reward_amount:
                    return total_rewards
                break

    return total_rewards

if __name__ == '__main__':
    env_name = 'CartPole-v1'  # gymnasium dùng 
    env = gym.make(env_name)

    rewards = vanilla_policy_gradient(env, PolicyEstimator(env), num_episodes=1500)

    moving_average_num = 100
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
    plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
    plt.title(f'Vanilla Policy Gradient on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
def play_trained_agent(env, estimator, episodes=5, render=True):
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
    
        while not done:
            if render:
                env.render()
                time.sleep(0.02)  # để chuyển động mượt hơn

            with torch.no_grad():
                action_probs = estimator.predict(obs).numpy()
            action = action_probs.argmax()  # chọn action xác suất cao nhất

            obs, reward, terminated, truncated, _ = env.step(action) ## trả về (next_state, reward, done, info)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1}: Total Reward = {total_reward}")

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="human")  # bật chế độ hiển thị
    estimator = PolicyEstimator(env)

    # Train agent
    rewards = vanilla_policy_gradient(env, estimator, num_episodes=500)

    # Chạy agent đã học để xem kết quả
    play_trained_agent(env, estimator, episodes=5, render=True)

    env.close()