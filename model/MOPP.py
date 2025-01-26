import torch 
import torch.distributions as dist

class MOPP:
    def __init__(self, env_dataset, q_networks, behavior_ensemble, dynamics_ensemble,
                 horizon, gamma, beta, kappa, Nm, L, num_trajectories, device, K_Q=10, batch_size=256):
        self.env_dataset = env_dataset
        self.q_networks = q_networks
        self.behavior_ensemble = behavior_ensemble
        self.dynamics_ensemble = dynamics_ensemble
        self.horizon = horizon
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.Nm = Nm
        self.L = L
        self.num_trajectories = num_trajectories
        self.device = device
        self.K_Q = K_Q  # Number of actions sampled for V_b(s_H)
        self.batch_size = batch_size
        self.optimized_actions = torch.zeros(self.horizon, behavior_ensemble.models[0].output_dim, device=device)

    def compute_uncertainty_matrix(self, trajectories):
        """
        Compute uncertainty matrix U for the state-action pairs in the trajectories.
        """
        U = []
        for trajectory in trajectories:
            uncertainty = []
            for (s, a, _, _) in trajectory:
                # Compute the discrepancy (or uncertainty) between dynamics model predictions
                predictions = []
                for model in self.dynamics_ensemble.models:
                    _, dynamics_output, _, _ = model.predict_dynamics(s, a)
                    predictions.append(dynamics_output)
                predictions = torch.stack(predictions)
                disc = torch.max(torch.norm(predictions - predictions.mean(dim=0), dim=-1))
                uncertainty.append(disc)
            U.append(uncertainty)
        return torch.tensor(U)

    def filter_trajectories(self, trajectories, uncertainty_matrix):
        """
        Filter trajectories based on the uncertainty matrix.
        """
        filtered_trajectories = []
        for idx, trajectory in enumerate(trajectories):
            uncertainties = uncertainty_matrix[idx]
            if torch.max(uncertainties) < self.L:
                filtered_trajectories.append(trajectory)
        
        # If not enough trajectories, select the ones with the lowest uncertainty
        if len(filtered_trajectories) < self.Nm:
            uncertainties_sum = [torch.sum(uncertainties) for uncertainties in uncertainty_matrix]
            sorted_idx = torch.argsort(torch.tensor(uncertainties_sum))
            for idx in sorted_idx[:self.Nm - len(filtered_trajectories)]:
                filtered_trajectories.append(trajectories[idx])

        return filtered_trajectories

    def compute_cumulative_return(self, trajectory, value_b_func):
        """
        Compute the cumulative return for a trajectory using the value function.
        """
        R_n = 0
        for t in range(self.horizon):
            R_n += self.gamma ** t * trajectory[t][3]  # Add reward at timestep t
        final_state = trajectory[-1][2]  # Terminal state
        R_n += self.gamma ** self.horizon * value_b_func(final_state)  # Add discounted terminal value
        return R_n

    def optimize_actions(self, pruned_trajectories, returns):
        action_dim = pruned_trajectories[0][0][1].shape[-1]
        weighted_actions = torch.zeros(self.horizon, action_dim).to(self.device)

        for t in range(self.horizon):
            weighted_sum = torch.zeros(action_dim).to(self.device)
            weight_sum = 0.0

            for trajectory, return_value in zip(pruned_trajectories, returns):
                return_tensor = torch.tensor(return_value, dtype=torch.float32).to(self.device)

                weight = torch.exp(torch.clamp(self.kappa * return_tensor, max=20))  # Prevent overflow in exp
                weight = weight.repeat(trajectory[t][1].shape[0], 1)

                weighted_sum += (weight * trajectory[t][1]).sum(dim=0)
                weight_sum += weight.sum()

            weighted_actions[t] = weighted_sum / (weight_sum + 1e-8)

        return weighted_actions


    def compute_value_b(self, q_networks, final_state, num_samples):
        """
        Compute the value function V_b(s_H) by averaging the Q-values for random action samples.
        
        Args:
            q_networks: List of Q-value networks.
            final_state: The terminal state.
            num_samples: Number of actions to sample for Q-value computation.
            
        Returns:
            The average Q-value for the terminal state.
        """
        # Sample random actions using behavior policy for the final state
        action_samples = [self.behavior_ensemble.predict_behavior(final_state)[0] for _ in range(num_samples)]
        
        q_values = []
        for action in action_samples:
            # Ensure that action is a tensor and move to the device
            if isinstance(action, tuple):
                action = action[0]  # Unpack if it's a tuple
            action = action.unsqueeze(0)  # Ensure action has shape [1, action_dim]
            
            # Flatten the action tensor to make sure it matches the state dimension
            action = action.view(-1, action.size(-1))  # Shape: [K_Q, action_dim]
    
            # Ensure final_state is also reshaped correctly
            final_state = final_state.squeeze(0)  # Remove the extra batch dimension, shape: [state_dim]
            final_state = final_state.unsqueeze(0).expand(action.size(0), -1)  # Shape: [K_Q, state_dim]
            
            # Compute Q-value using the q_networks
            q_value = q_networks[0](final_state, action)  # Assuming q_networks is a list of Q-networks
            q_values.append(q_value)
        
        # Average the Q-values
        q_values = torch.stack(q_values)
        value_b = q_values.mean()
        
        return value_b

    def train(self, state):
        dim_1 = self.horizon + 1
        # Step 1: Initialization and train models (already done)
        A_star = torch.zeros(dim_1, self.behavior_ensemble.models[0].output_dim, device=self.device)  # shape: [3,6]
    
        # Step 2: Begin infinite loop
        for tau in range(1):
            # Step 3: Observe and initialize
            s_tau = state  # Use the provided state, shape: [1,17]
            s_tau = s_tau.to(self.device)  # Ensure the state is on the correct device
            R = []  # list to store rewards
            T = []  # Initialize trajectory
    
            # Step 4: Iterate over trajectories
            for n in range(self.num_trajectories):
                # Step 5: Initialize quantities
                s_0 = s_tau  # initial state for trajectory
                R_n = 0
                T_n = []

                # Step 6: Iterate over timesteps in the trajectories
                for t in range(self.horizon):
                    # Step 7: Sample action a_t using behavior policy
                    model_index_beh = torch.randint(0, len(self.behavior_ensemble.models), (1,)).item()
                    behavior_model = self.behavior_ensemble.models[model_index_beh].to(self.device)
                    mean_actions, _, _, _ = behavior_model.predict_behavior(s_tau)
    
                    sigma_a = torch.abs(mean_actions)  # Get standard deviation (uncertainty)
                    sigma_a = torch.clamp(sigma_a, min=1e-8)  # Avoid zero or negative values
    
                    # Create a Normal distribution based on mean and std
                    distr = dist.Normal(mean_actions, sigma_a)
                    action_samples = distr.sample((self.K_Q,))  # Sample K_Q actions, shape: [K_Q, 1, action_dim]
                    
                    #print("action_samples distr", action_samples.shape)
                    action_samples = action_samples.squeeze(1)  # Shape: [K_Q, action_dim]
                    #print("action samples squeeze", action_samples.shape)
                    
                    state_expanded = s_tau.unsqueeze(0).expand(self.K_Q, -1, -1)  # Shape: [K_Q, batch_size, state_dim]
                    #print("state_expanded expand", state_expanded.shape)
                    state_expanded = state_expanded.contiguous().view(-1, state_expanded.size(-1))  # Shape: [K_Q * batch_size, state_dim]
                    #print("state_expanded view", state_expanded.shape)
                     
                    # Step 8: Compute Q-values for all sampled actions
                    q_values = []
                    for q_network in self.q_networks:
                        # Pass state and action separately to the Q-network
                        q_value = q_network(state_expanded, action_samples)  # Compute Q-value
                        q_values.append(q_value.view(self.K_Q, -1))  # Reshape back to [K_Q, batch_size]
                    
                    q_values = torch.stack(q_values, dim=0)  # Shape: [num_q_networks, K_Q, batch_size]
                    #print("q_values", q_values.shape)
    
                    # Compute the mean Q-value across networks and find the best action
                    mean_q_values = q_values.mean(dim=0)  # Average over the Q-networks, shape: [K_Q, batch_size]
                    _, best_action_indices = torch.max(mean_q_values, dim=0)  # Shape: [batch_size]
                    #print("best_action_indices", best_action_indices)
                    action_t_hat = action_samples[best_action_indices, range(best_action_indices.size(0))]  # Optimal action
                    
                    # Compute action_t_tilde
                    A_star_t_plus_1 = A_star[t + 1].unsqueeze(0).expand(action_t_hat.size(0), -1)  # Shape: [batch_size, action_dim]
                    action_t_tilde = (1 - self.beta) * action_t_hat + self.beta * A_star_t_plus_1
    
                    # Step 9: Append (s_t, a_t) to trajectory
                    T_n.append((s_tau, action_t_tilde))
    
                    # Step 10: Predict next state using dynamics model
                    model_index_dyn = torch.randint(0, len(self.dynamics_ensemble.models), (1,)).item()
                    dynamics_model = self.dynamics_ensemble.models[model_index_dyn].to(self.device)
                    _, dynamics_output, _, _ = dynamics_model.predict_dynamics(s_0, action_t_tilde)
                    next_state = dynamics_output[:, :s_0.size(-1)]
                    reward = dynamics_output[:, s_0.size(-1):]
    
                    # Update trajectory and state
                    T_n[-1] = (s_0.clone(), action_t_tilde.clone(), next_state.clone(), reward.clone())
                    s_0 = next_state.clone()
    
                    # Compute the reward using all dynamics models
                    rewards = []
                    for dynamics_model in self.dynamics_ensemble.models:
                        dynamics_model = dynamics_model.to(self.device)
                        _, dynamics_output, _, _ = dynamics_model.predict_dynamics(s_0, action_t_tilde)
                        rewards.append(dynamics_output[:, s_0.size(-1):])

                    # Average the rewards across all dynamics models
                    mean_reward = torch.stack(rewards).mean(dim=0)
                    R_n += mean_reward.mean().item()
    
                # Evaluate terminal value Vb(sH)
                final_state = T_n[-1][2]  # s_H
                value_b = self.compute_value_b(self.q_networks, final_state, num_samples=self.K_Q)
    
                # Store trajectory and rewards
                T.append(T_n)
                R_n += value_b.item()
                R.append(R_n)
    
            # Compute uncertainty matrix
            uncertainty_matrix = self.compute_uncertainty_matrix(T)
    
            # Prune trajectories based on uncertainty
            pruned_trajectories = self.filter_trajectories(T, uncertainty_matrix)
    
            # Optimize action sequence
            self.optimized_actions = self.optimize_actions(pruned_trajectories, R)
    
        # Return optimized action for the initial state
        return self.optimized_actions[0]