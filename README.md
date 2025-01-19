# Model-Based Offline Planning (MOPP)
This repository implements Model-Based Offline Planning (MOPP), a cutting-edge algorithm for Offline Reinforcement Learning (Offline RL). MOPP addresses the challenges of leveraging static datasets to optimize actions while avoiding out-of-distribution (OOD) errors. It integrates dynamics models, Q-value networks, and trajectory optimization to achieve efficient decision-making in simulated environments.

## Key Features
- **Autoregressive Dynamics Model** (ADM): captures probabilistic dynamics and behavior policy using a flexible autoregressive decomposition.
- **Q-Value Network**: evaluates state-action pairs to guide trajectory rollouts toward high-reward regions.
- **Trajectory Optimization and Pruning**: efficiently generates and refines action sequences to maximize rewards while managing uncertainty.

## Environments
This project uses the D4RL MuJoCo dataset for benchmarking. The supported environments include:
1. **HalfCheetah**: A 2D quadruped with the goal of running.
2. **Walker2D**: A 2D biped with the goal of walking.
3. **Hopper**: A 2D monoped with the goal of hopping.

These environments provide static datasets for offline RL research.