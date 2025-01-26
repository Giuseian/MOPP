import os
import numpy as np 
import torch 
import json


# Load pretrained dynamics and behavior models
def load_pretrained_ensemble(ensemble, pretrained_paths, optimizers, device):
    for i, (model, optimizer) in enumerate(zip(ensemble.models, optimizers)):
        model.to(device)
        if pretrained_paths[i] is not None:  # Check if a path is provided
            checkpoint = torch.load(pretrained_paths[i], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded pretrained model {i + 1} from {pretrained_paths[i]}")
        else:
            print(f"No pretrained model provided for model {i + 1}. Training from scratch.")


def initialize_ensemble_with_pretrained(ensemble, pretrained_paths, optimizers, device):
    for i, (model, optimizer, pretrained_path) in enumerate(zip(ensemble.models, optimizers, pretrained_paths)):
        model.to(device)
        if pretrained_path is not None:  # If pretrained path is provided
            checkpoint = torch.load(pretrained_path, map_location=device)  # Load the checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Pretrained model {i + 1} loaded from {pretrained_path}.")
        else:
            print(f"No pretrained model provided for model {i + 1}. Training from scratch.")


def evaluate_value_function_ensemble(q_networks, state, behavior_policies, num_samples=10):
    """
    Evaluates the value function V_b(s) using an ensemble of Q-networks and behavior policies.

    Args:
        q_networks: List of Q-value networks.
        state: State tensor for evaluation.
        behavior_policies: List of behavior policy models.
        num_samples: Number of actions to sample from each behavior policy.

    Returns:
        The average value function V_b(s) over all ensembles.
    """
    total_value = 0.0
    for q_network, behavior_policy in zip(q_networks, behavior_policies):
        for _ in range(num_samples):
            _, action, _, _ = behavior_policy.predict_behavior(state)
            total_value += q_network(state, action).item()

    return total_value / (len(q_networks) * num_samples)  # Average over all models and samples


def perform_max_q_operation(q_networks, behavior_policies, state, num_samples=10):
    """
    Performs the max-Q operation using an ensemble of Q-networks and behavior policies.

    Args:
        q_networks: List of Q-value networks.
        behavior_policies: List of behavior policy models.
        state: State tensor for evaluation.
        num_samples: Number of actions to sample from each behavior policy.

    Returns:
        Action with the maximum Q-value across the ensemble.
    """
    max_q_value = float("-inf")
    best_action = None

    for q_network, behavior_policy in zip(q_networks, behavior_policies):
        for _ in range(num_samples):
            _, action, _, _ = behavior_policy.predict_behavior(state)
            q_value = q_network(state, action).item()

            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action

    return best_action


def eval_policy_episodes(env, policy, n_episodes, device):
    results = []

    for i in range(n_episodes):
        print(f"Episode {i + 1}")

        # env.reset()
        # Start with the initial state from the dataset
        state = env.observations[i].to(device).unsqueeze(0)  # Ensure correct shape
        #print("Initial state:", state.shape)
        cumulative_reward = 0.0

        # Get precomputed actions for this episode
        #optimal_actions = precomputed_actions[i]
        #print(f"Optimal actions shape: {optimal_actions.shape}")
        optimal_actions = policy.train(state)
        #print("optimal_actions", optimal_actions)

        # Horizon-based loop
        for t in range(policy.horizon):
            # Use the precomputed action
            action = optimal_actions[t].detach().cpu().numpy()
            #print(f"Step {t + 1}, Action shape: {action.shape}")

            # Step the environment
            next_state, reward, done = env.step(action)
            cumulative_reward += reward

            # Convert next_state to a tensor and update state
            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            if done:
                print(f"Episode terminated early at step {t + 1}")
                break

        results.append(cumulative_reward)
        print(f"Completed Episode {i + 1}! Total Reward: {cumulative_reward:.2f}")

    results = np.array(results)
    return float(np.mean(results)), float(np.std(results)), results


def save_results(results, save_dir, save_format="json"):
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results.{save_format}"
    file_path = os.path.join(save_dir, filename)

    if save_format == "json":
        # Convert numpy array to list for JSON serialization
        results_data = {
            "mean_reward": float(np.mean(results)),
            "std_reward": float(np.std(results)),
            "rewards": results.tolist(),
        }
        with open(file_path, "w") as f:
            json.dump(results_data, f, indent=4)
    else:
        raise ValueError("Invalid save format. Use 'json' or 'pickle'.")

    print(f"Results saved to {file_path}")