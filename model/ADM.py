import numpy as np 
import torch 
import torch.nn as nn
import torch.distributions as dist
import torch.distributions as dist

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ADM(nn.Module):
    """Unified ADM network: Supports both behavior policy and dynamics model"""
    def __init__(self, output_dim, input_dim, fc_layer_params=(), out_reranking=None, latent_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.out_rank = np.arange(output_dim) if out_reranking is None else out_reranking
        latent_dim = latent_dim or output_dim * 2
        self.shared_layer, self._layers_list = self._initialize_layers(fc_layer_params, latent_dim)

    def _initialize_layers(self, fc_layer_params, latent_dim):
        shared_layer, layers_list = nn.ModuleList(), nn.ModuleList()
        if isinstance(fc_layer_params[0], (list, tuple)):
            shared_params, indiv_params = fc_layer_params
            # Initialize shared layers
            for i, n in enumerate(shared_params):
                in_features = self.input_dim if i == 0 else shared_params[i - 1]
                shared_layer.append(nn.Linear(in_features, n))
            shared_layer.append(nn.Linear(shared_params[-1], latent_dim))
        else:
            indiv_params = fc_layer_params
    
        # Initialize individual output layers
        for _ in range(self.output_dim):
            layers = nn.ModuleList()
            for i, n in enumerate(indiv_params):
                # Adjust input size for concatenated tensor (latent_dim + mode.size)
                in_features = latent_dim + _ if i == 0 else indiv_params[i - 1]
                layers.append(nn.Linear(in_features, n))
            layers.append(nn.Linear(indiv_params[-1], 2))  # Mean and std
            layers_list.append(layers)
    
        return shared_layer, layers_list

    def _forward_layers(self, x, layers):
        for layer in layers:
            x = torch.relu(layer(x))
        return x

    def _get_outputs(self, inputs, layers):
        h = self._forward_layers(inputs, layers)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        distribution = dist.Normal(mean, std)
        return distribution, mean

    def _update_tensor(self, a, b, index):
        # If `b` is scalar, unsqueeze to make it (1, 1)
        if b.dim() == 0:  # If `b` is scalar, make it (1, 1)
            b = b.unsqueeze(0).unsqueeze(1)  # Reshape to (1, 1)
        elif b.dim() == 1:  # If `b` is 1D, add another dimension
            b = b.unsqueeze(1)  # Reshape to (batch_size, 1)
    
        # Ensure `b` has the same batch size as `a`
        if b.size(0) != a.size(0):
            raise ValueError(f"Mismatch in batch size: a has {a.size(0)} rows, but b has {b.size(0)} rows.")
    
        # Ensure `b` has the correct dimensions (2D tensor with size [batch_size, 1])
        if b.dim() != 2 or b.size(1) != 1:
            raise ValueError(f"Tensor `b` must have 2 dimensions and size (batch_size, 1). Got {b.shape}.")
    
        # Split `a` into left and right parts
        left = a[:, :index]
        right = a[:, index + 1:]
    
        # Concatenate left, b, and right along dimension 1
        return torch.cat([left, b, right], dim=1)


    def forward(self, inputs):
        h = inputs
        if self.shared_layer:
            for layer in self.shared_layer:
                h = torch.relu(layer(h))
    
        batch_size = inputs.size(0)
        out_mean = torch.zeros(batch_size, self.output_dim, device=inputs.device)
        outs_sample = torch.zeros(batch_size, self.output_dim, device=inputs.device)
        log_pi_outs = torch.zeros(batch_size, self.output_dim, device=inputs.device)
        outs_dist = [None] * self.output_dim
    
        for i, index in enumerate(self.out_rank):
            dist, mode = self._get_outputs(h, self._layers_list[i])
    
            # Ensure mode has the shape [batch_size, 1] (2D tensor)
            if mode.dim() == 3:  # If mode has shape [batch_size, 1, 1], squeeze the extra dimension
                mode = mode.squeeze(2)  # This will make it [batch_size, 1]
            elif mode.dim() == 1:  # If mode is [batch_size], we need to add an extra dimension
                mode = mode.unsqueeze(1)  # This will make it [batch_size, 1]
    
            # Now, h is [batch_size, feature_size] and mode is [batch_size, 1]
            #print(f"h.shape: {h.shape}")  # Debug print
            #print(f"mode.shape: {mode.shape}")  # Debug print
    
            # Concatenate h and mode
            h = torch.cat([h, mode], dim=-1)  # This should now work
    
            sample = dist.rsample()
            log_prob = dist.log_prob(sample)
    
            out_mean = self._update_tensor(out_mean, mode, index)
            outs_sample = self._update_tensor(outs_sample, sample, index)
            log_pi_outs = self._update_tensor(log_pi_outs, log_prob, index)
            outs_dist[index] = dist
    
        return out_mean, outs_sample, log_pi_outs, outs_dist


    def predict_behavior(self, state):
        """Predicts behavior policy given the state."""
        return self.forward(state)

    def predict_dynamics(self, state, action):
        """Predicts next state and reward given state and action."""
        inputs = torch.cat([state, action], dim=-1)
        return self.forward(inputs)