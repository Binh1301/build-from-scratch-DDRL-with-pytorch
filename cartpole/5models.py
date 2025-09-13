# train_compare_sb3_gymnasium.py

import os, time
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import A2C, PPO, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

OUT_DIR = "./sb3_experiments"
os.makedirs(OUT_DIR, exist_ok=True)

# Config: chạy cả 5 mô hình trên CartPole-v1
env_configs = {
    "cartpole": {
        "id": "CartPole-v1",
        "algos": ["A2C", "PPO", "SAC", "DDPG", "TD3"],
        "timesteps": 50000
    }
}

def make_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return Monitor(env)
    return _init

class SaveLogsCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.rewards, self.lengths, self.times = [], [], []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.rewards.append(ep["r"])
                self.lengths.append(ep["l"])
                self.times.append(time.time())
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame({
            "reward": self.rewards,
            "length": self.lengths,
            "timestamp": self.times
        })
        df.to_csv(self.save_path, index=False)

def train_model(algo_name, env_id, total_timesteps):
    run_id = f"{algo_name}_{env_id}_{int(time.time())}"
    csv_path = os.path.join(OUT_DIR, run_id + ".csv")
    model_path = os.path.join(OUT_DIR, run_id + ".zip")

    vec_env = DummyVecEnv([make_env(env_id, seed=0)])
    vec_env = VecMonitor(vec_env)
    cb = SaveLogsCallback(csv_path)

    if algo_name == "A2C":
        model = A2C("MlpPolicy", vec_env, verbose=1)
    elif algo_name == "PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1)
    elif algo_name == "SAC":
        model = SAC("MlpPolicy", vec_env, verbose=1)
    elif algo_name == "DDPG":
        model = DDPG("MlpPolicy", vec_env, verbose=1)
    elif algo_name == "TD3":
        model = TD3("MlpPolicy", vec_env, verbose=1)
    else:
        raise ValueError("Unknown algo")

    print(f"[TRAIN] {algo_name} on {env_id} for {total_timesteps} timesteps.")
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(model_path)
    vec_env.close()
    return csv_path, model_path

def aggregate_and_plot(results):
    all_dfs = []
    for r in results:
        if os.path.exists(r["log"]):
            df = pd.read_csv(r["log"])
            df["algo"], df["env"] = r["algo"], r["env"]
            if len(df) > 0:
                df["reward_smooth"] = df["reward"].rolling(window=10, min_periods=1).mean()
            all_dfs.append(df)

    agg_path = os.path.join(OUT_DIR, "aggregated_logs.csv")
    if all_dfs:
        agg = pd.concat(all_dfs, ignore_index=True)
        agg.to_csv(agg_path, index=False)
        print(f"Saved aggregated logs to {agg_path}")

    # Vẽ tất cả mô hình trên cùng 1 hình
    plt.figure(figsize=(10, 6))
    plotted = False
    for r in results:
        if not os.path.exists(r["log"]):
            continue
        df = pd.read_csv(r["log"])
        if df.empty:
            continue
        df['episode_idx'] = np.arange(len(df))
        df['reward_smooth'] = df['reward'].rolling(window=10, min_periods=1).mean()
        plt.plot(df['episode_idx'], df['reward_smooth'], label=r['algo'])
        plotted = True

    if plotted:
        plt.title("Training curves (smoothed) on CartPole-v1")
        plt.xlabel("Episode")
        plt.ylabel("Episode reward (smoothed)")
        plt.legend()
        png = os.path.join(OUT_DIR, "comparison_cartpole.png")
        plt.savefig(png)
        plt.show()
        print(f"Saved plot {png}")

def main():
    results = []
    for key, cfg in env_configs.items():
        for algo in cfg["algos"]:
            try:
                csv_path, model_path = train_model(algo, cfg["id"], cfg["timesteps"])
                results.append({
                    "algo": algo,
                    "env": cfg["id"],
                    "log": csv_path,
                    "model": model_path
                })
            except Exception as e:
                print(f"Training failed for {algo} on {cfg['id']}: {e}")
    aggregate_and_plot(results)
    print("All done. Check the folder:", OUT_DIR)

if __name__ == "__main__":
    main()
