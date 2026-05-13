"""
Yuli Tshuva
Create a model for sequence similarity based on the graph distance algorithm.
"""

# Imports
import torch
import torch.nn as nn
from torch.ao.nn.quantized.functional import threshold
from torch.nn.parameter import Parameter
from torch.optim import Adam
from tqdm.auto import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR, MIN_LR = 1, 1e-5
PATIENCE = 10
EPOCHS = 5000

import torch
import torch.nn as nn
from torch.nn import Parameter


class SequenceSimilarity(nn.Module):
    def __init__(self, initial_match_matrix, features1, features2, alpha, gamma):
        super(SequenceSimilarity, self).__init__()

        # 1. Store alpha and gamma as buffers (not learnable parameters)
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))

        # 2. Learnable Match Matrix in Log-Space
        epsilon = 1e-8
        log_init = torch.log(torch.tensor(initial_match_matrix, dtype=torch.float32) + epsilon)
        self.sigma_logits = Parameter(log_init)

        # 3. Features as buffers
        f1 = torch.tensor(features1, dtype=torch.float32)
        f2 = torch.tensor(features2, dtype=torch.float32)
        self.register_buffer('features1', f1)
        self.register_buffer('features2', f2)

        # 4. Pre-compute Feature Distance Matrix (fixed, since features are constant)
        self.register_buffer('feat_dist_matrix', torch.cdist(f1, f2, p=2) ** 2)

        # 5. Pre-compute Structural Distance Buffers
        n1, n2 = features1.shape[0], features2.shape[0]
        idx1 = torch.arange(n1, dtype=torch.float32)
        idx2 = torch.arange(n2, dtype=torch.float32)
        D1 = torch.abs(idx1.unsqueeze(1) - idx1.unsqueeze(0))
        D2 = torch.abs(idx2.unsqueeze(1) - idx2.unsqueeze(0))
        self.register_buffer('D1', D1)
        self.register_buffer('D2', D2)

        # 6. Pre-compute Index Proximity Distance Buffer (assumes n1 == n2)
        n = n1
        idx = torch.arange(n, dtype=torch.float32)
        self.register_buffer('index_dist', (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2)

    def get_constrained_sigma(self):
        sigma = torch.exp(self.sigma_logits) * (self.sigma_logits > -4)
        sigma = sigma / (sigma.sum(dim=1, keepdim=True) + 1e-9)
        return sigma

    def compute_features_distance(self, sigma):
        # feat_dist_matrix is pre-computed, so this is now a simple weighted sum
        return torch.sum(sigma * self.feat_dist_matrix)

    def compute_structure_distance_vectorized(self, sigma):
        row_sums = sigma.sum(dim=1)
        col_sums = sigma.sum(dim=0)

        row_weights = row_sums.unsqueeze(1) * row_sums.unsqueeze(0)
        term_a = torch.sum(row_weights * (self.D1 ** 2))

        col_weights = col_sums.unsqueeze(1) * col_sums.unsqueeze(0)
        term_b = torch.sum(col_weights * (self.D2 ** 2))

        inter_matrix = torch.matmul(torch.matmul(sigma.t(), self.D1), sigma)
        term_c = -2 * torch.sum(inter_matrix * self.D2)

        return term_a + term_b + term_c

    def compute_index_proximity_cost(self, sigma):
        return torch.sum(sigma * self.index_dist)

    def forward(self):
        sigma = self.get_constrained_sigma()

        f_dist   = self.compute_features_distance(sigma)
        s_dist   = self.compute_structure_distance_vectorized(sigma)
        idx_dist = self.compute_index_proximity_cost(sigma)

        return f_dist, self.alpha, s_dist, self.gamma, idx_dist
def train_model(model, save_loss=False):
    # Set the optimizer
    lr = LR
    optimizer = Adam(model.parameters(), lr=lr)
    # Set a scheduler to reduce the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Set a variable for the model's best state
    best_model_state = None

    # Set a variable to track the best loss and the number of epochs without improvement
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Optionally track loss history
    if save_loss:
        f_loss, s_loss = [], []
        scheduler_steps = []

    # Train the model
    model.train()
    for epoch in range(EPOCHS):
        # Calculate the loss
        model_output = model()
        match_loss = model_output[0] + model_output[1] * model_output[2]
        # Backpropagation
        optimizer.zero_grad()
        match_loss.backward()
        optimizer.step()

        # Add loss to history if needed
        if save_loss:
            f_loss.append(model_output[0].detach().numpy())
            s_loss.append(model_output[2].detach().numpy())

        # Check for improvement
        if match_loss.item() < 0.99 * best_loss:
            best_loss = match_loss.item()
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # If no improvement for a certain number of epochs, stop training
        if epochs_without_improvement >= PATIENCE:
            if lr < MIN_LR:
                break
            else:
                if save_loss:
                    scheduler_steps.append(epoch)
                scheduler.step()
                lr *= 0.1
                epochs_without_improvement = 0

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if save_loss:
        return model, best_loss, f_loss, s_loss, scheduler_steps
    return model, best_loss
