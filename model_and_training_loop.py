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
    def __init__(self, initial_match_matrix, features1, features2, alpha):
        super(SequenceSimilarity, self).__init__()
        # Store alpha and beta as buffers (not learnable parameters)
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        # 1. Target Sums (Fixed Constraints) - Stored as buffers
        row_sums = torch.sum(torch.tensor(initial_match_matrix, dtype=torch.float32), dim=1)
        col_sums = torch.sum(torch.tensor(initial_match_matrix, dtype=torch.float32), dim=0)
        self.register_buffer('target_row_sums', row_sums)
        self.register_buffer('target_col_sums', col_sums)

        # 2. Learnable Match Matrix in Log-Space
        # We use Log-space so that when we apply exp() in forward,
        # the values are strictly positive, which prevents division by zero.
        # Adding epsilon prevents log(0)
        epsilon = 1e-8
        log_init = torch.log(torch.tensor(initial_match_matrix, dtype=torch.float32) + epsilon)
        self.sigma_logits = Parameter(log_init)

        # 3. Features as buffers
        self.register_buffer('features1', torch.tensor(features1, dtype=torch.float32))
        self.register_buffer('features2', torch.tensor(features2, dtype=torch.float32))

        # 4. Pre-compute Structural Distance Buffers
        n1, n2 = features1.shape[0], features2.shape[0]
        idx1 = torch.arange(n1, dtype=torch.float32)
        idx2 = torch.arange(n2, dtype=torch.float32)
        D1 = torch.abs(idx1.unsqueeze(1) - idx1.unsqueeze(0))
        D2 = torch.abs(idx2.unsqueeze(1) - idx2.unsqueeze(0))
        self.register_buffer('D1', D1)
        self.register_buffer('D2', D2)

    def get_constrained_sigma(self):
        sigma = torch.exp(self.sigma_logits) * torch.tensor(self.sigma_logits > -4, dtype=torch.float32)

        for _ in range(5):
            # Normalize Columns to match the SPECIFIC target_col_sums
            col_sums = sigma.sum(dim=0, keepdim=True)
            sigma = sigma * (self.target_col_sums / (col_sums + 1e-9))

            # Normalize Rows to match the SPECIFIC target_row_sums
            row_sums = sigma.sum(dim=1, keepdim=True)
            sigma = sigma * (self.target_row_sums.unsqueeze(1) / (row_sums + 1e-9))

        return sigma

    def compute_features_distance(self, sigma):
        # Efficient squared Euclidean distance
        dist_matrix = torch.cdist(self.features1, self.features2, p=2) ** 2
        return torch.sum(sigma * dist_matrix)

    def compute_structure_distance_vectorized(self, sigma):
        # Term A & B (Constants based on target sums)
        row_weights = self.target_row_sums.unsqueeze(1) * self.target_row_sums.unsqueeze(0)
        term_a = torch.sum(row_weights * (self.D1 ** 2))

        col_weights = self.target_col_sums.unsqueeze(1) * self.target_col_sums.unsqueeze(0)
        term_b = torch.sum(col_weights * (self.D2 ** 2))

        # Term C (Interaction)
        inter_matrix = torch.matmul(torch.matmul(sigma.t(), self.D1), sigma)
        term_c = -2 * torch.sum(inter_matrix * self.D2)

        return term_a + term_b + term_c

    def forward(self):
        # 1. Generate the normalized sigma (DO NOT re-assign self.sigma_logits)
        sigma = self.get_constrained_sigma()

        # 2. Compute distances
        f_dist = self.compute_features_distance(sigma)
        s_dist = self.compute_structure_distance_vectorized(sigma)

        if s_dist < 0:
            pass

        # 3. Combine
        return f_dist + self.alpha * s_dist


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
        loss_history = []
        scheduler_steps = []

    # Train the model
    model.train()
    for epoch in range(EPOCHS):
        # Calculate the loss
        match_loss = model()
        # Backpropagation
        optimizer.zero_grad()
        match_loss.backward()
        optimizer.step()

        # Add loss to history if needed
        if save_loss:
            loss_history.append(match_loss.item())

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
        return model, best_loss, loss_history, scheduler_steps
    return model, best_loss
