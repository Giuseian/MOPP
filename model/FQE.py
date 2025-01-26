import torch
import os


def train_q_value_network_ensemble(q_networks, target_q_networks, behavior_policies, dataloader, optimizers, gamma=0.99, num_epochs=40, device="cuda"):
    # Move models to the correct device
    for q_network in q_networks:
        q_network.to(device)
    for target_q_network in target_q_networks:
        target_q_network.to(device)
    for behavior_policy in behavior_policies:
        behavior_policy.to(device)

    for epoch in range(num_epochs):
        epoch_losses = [0.0 for _ in range(len(q_networks))]
        for batch_idx, (state, action, next_state, reward) in enumerate(dataloader):
            # Move data to the device
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            reward = reward.to(device)

            for i, (q_network, target_q_network, behavior_policy, optimizer) in enumerate(zip(q_networks, target_q_networks, behavior_policies, optimizers)):
                optimizer.zero_grad()

                # Compute current Q-value
                current_q_value = q_network(state, action)

                with torch.no_grad():
                    # Predict next action using the behavior policy
                    _, next_action, _, _ = behavior_policy.predict_behavior(next_state)
    
                    # Compute target Q-value
                    target_q_value = reward.unsqueeze(-1) + gamma * target_q_network(next_state, next_action)

                # Compute loss
                loss = torch.nn.functional.mse_loss(current_q_value, target_q_value)

                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_losses[i] += loss.item()

        # Print losses
        for i, loss in enumerate(epoch_losses):
            print(f"Epoch {epoch + 1}/{num_epochs}, Q-Network {i + 1} Loss: {loss:.4f}")