import gym
import torch
from torch.utils.data import Dataset 

class Environment(Dataset):
    def __init__(self, env_name, normalization_states=False, normalization_rewards=False, subset_size=None):
        self.env = gym.make(env_name)
        self.dataset = self.env.get_dataset()  # Load the D4RL dataset

        # Extract data from the D4RL dataset
        self.observations = torch.tensor(self.dataset['observations'], dtype=torch.float32)
        self.actions = torch.tensor(self.dataset['actions'], dtype=torch.float32)
        self.rewards = torch.tensor(self.dataset['rewards'], dtype=torch.float32)
        self.next_states = torch.tensor(self.dataset['next_observations'], dtype=torch.float32)
        self.dones = torch.tensor(self.dataset['terminals'], dtype=torch.float32)

        # Subset the dataset if subset_size is provided
        if subset_size is not None:
            self.observations = self.observations[:subset_size]
            self.actions = self.actions[:subset_size]
            self.rewards = self.rewards[:subset_size]
            self.next_states = self.next_states[:subset_size]
            self.dones = self.dones[:subset_size]

        # Normalize data if required
        if normalization_states:
            self.normalize_states()
        if normalization_rewards:
            self.normalize_rewards()

        # Initialize for episodic simulation
        self.num_steps = len(self.observations)
        self.current_idx = 0 

    def normalize_states(self):
        self.shift = -self.observations.mean(dim=0)
        self.scale = 1.0 / (self.observations.std(dim=0) + 1e-3)

        # Normalize observations and next_states
        self.observations = (self.observations + self.shift) * self.scale
        self.next_states = (self.next_states + self.shift) * self.scale

    def normalize_rewards(self):
        self.r_max = self.rewards.max()
        self.r_min = self.rewards.min()
        self.rewards = (self.rewards - self.r_min) / (self.r_max - self.r_min)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'done': self.dones[idx]
        }

    def reset(self):
        self.current_idx = 0
        return self.observations[self.current_idx]
    
    def step(self, action):
        # Taking a step in the environment 
        #print("self.current_idx", self.current_idx)
        if self.current_idx >= self.num_steps:
            raise IndexError("Dataset Exhausted. Call reset() to restart")

        observation = self.observations[self.current_idx]
        reward = self.rewards[self.current_idx]
        next_state = self.next_states[self.current_idx]
        done = self.dones[self.current_idx]

        self.current_idx += 1  # Move to the next step
        return next_state, reward, done
