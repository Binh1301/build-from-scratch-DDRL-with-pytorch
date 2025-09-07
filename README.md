README — Deep Deterministic Reinforcement Learning (DDRL) from scratch with PyTorch

Project: Implementing Deep Deterministic RL algorithms from scratch in PyTorch, with training logs and visualization for models, losses, and network parameters.
Goal: Provide a clean, well-documented, reproducible repository for learning and experimenting with deterministic continuous-control RL (e.g., DDPG / TD3 style methods), including tools to visualize training, networks and saved parameters.

🚀 Project overview

This repository contains a minimal, well-commented implementation of Deep Deterministic Reinforcement Learning (DDRL) algorithms written from scratch in PyTorch. It focuses on clarity and reproducibility so you can:

Understand and experiment with actor–critic deterministic policies (DDPG / TD3-style).

Log scalar metrics, network weights and gradients.

Visualize training progress (scores, losses, action distributions).

Save and load checkpoints for models and optimizer states.

Note: “DDRL” here refers to deterministic continuous-action deep RL approaches (DDPG / TD3 family). You can easily extend the code to other algorithms (SAC, PPO, etc.).

🧩 Features

Pure PyTorch implementation (no RL frameworks): actor, critic, target networks, replay buffer, soft updates.

Configurable hyperparameters in a single config file / class.

Training loop with reproducible seeding.

Logging:

CSV logs (episode, reward, losses, lr, etc.)

TensorBoard logging (scalars, histograms, images)

Optional Weights & Biases integration (W&B)

Visualizations:

Episode rewards, running average.

Actor / critic loss plots.

Network weight & gradient histograms over time.

Action distribution scatter/hist plots and state-action visualization.

Checkpointing: save/restore model & optimizer & RNG seeds.

Example training script for Pendulum-v1 (or any Gym/Gymnasium env with continuous actions).
