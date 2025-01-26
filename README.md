# Model-Based Offline Planning (MOPP)
This repository implements Model-Based Offline Planning (MOPP), a cutting-edge algorithm for Offline Reinforcement Learning (Offline RL). MOPP addresses the challenges of leveraging static datasets to optimize actions while avoiding out-of-distribution (OOD) errors. It integrates dynamics models, Q-value networks, and trajectory optimization to achieve efficient decision-making in simulated environments.

## Overview
Offline RL is a branch of reinforcement learning where the policy is learned from a static, pre-collected dataset of environment interactions, without additional interactions with the environment. A key challenge in Offline RL is keeping policy learning close to the data distribution. Taking actions that deviate from the dataset (OOD samples) can lead to inaccurate predictions by the learned models, causing suboptimal or unsafe decisions. To address this, Model-Based Planning builds a dynamics model of the environment, which predicts how the environment will respond to actions. 

Our work focuses on implementing a Model-Based Offline Planning algorithm. Specifically, it conducts trajectory rollouts guided by a behavior policy learned directly from data. The algorithm identifies and prunes problematic trajectories to minimize out-of-distribution (OOD) actions, improving 
decision-making and optimizing outcomes.

### Key Features
- **Autoregressive Dynamics Model** (ADM): It learns the environmet’s behavior by predicting its response to different actions.
- **Q-Value Network**: Guides the policy to estimate the value of various state-action pairs
- **Trajectory Optimization and Pruning**: Generates and refines action sequences to maximize rewards while managing uncertainty.

## Environments
This project uses the D4RL MuJoCo dataset for benchmarking. The supported environments include:
1. **HalfCheetah**: A 2D quadruped with the goal of running.
2. **Walker2D**: A 2D biped with the goal of walking.
3. **Hopper**: A 2D monoped with the goal of hopping.

These environments provide static datasets for offline RL research. They are all stochastic in terms of their initial state, with a Gaussian noise added to a fixed initial state in order to add stochasticity.

### Set-up
Environment set-up extends PyTorch's Dataset and integrates it with Gym environments to load and preprocess offline datasets (e.g., D4RL). Key features include: 
1. **Data Loading & Preprocessing**: Loads observations, actions, rewards, next states, and done flags from the dataset. Supports dataset subsetting for efficient training and testing.
2. Optional **normalization** for states and rewards to enhance model stability and learning efficiency. State Normalization shifts and scales observations and next states using dataset statistics. Reward Normalization rescales rewards to [0, 1] for consistent training objectives.
3. The **reset** method is called at the start of each episode to initialize the environment.
4. The **step** method is called repeatedly during an episode to sample transitions until the done flag is true or the dataset is exhausted.

## Proposed Method