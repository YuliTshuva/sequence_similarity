"""
Yuli Tshuva
Create a model for sequence similarity based on the graph distance algorithm.
"""

# Imports
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR, MIN_LR = 1, 1e-6
PATIENCE = 50
EPOCHS = 5000


class SequenceSimilarity(nn.Module):
    def __init__(self, initial_match_matrix, features1, features2, alpha):
        super(SequenceSimilarity, self).__init__()

        epsilon = 1e-8

        # 1. Store alpha/beta as buffers (not learnable)
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        # 2. Learnable Match Matrix in Log-Space — now n1 x n2
        log_init = torch.log(torch.tensor(initial_match_matrix, dtype=torch.float32) + epsilon)
        self.sigma_logits = Parameter(log_init)  # shape: (n1, n2)

        # 3. Features as buffers — potentially different lengths
        f1 = torch.tensor(features1, dtype=torch.float32)  # (n1, d)
        f2 = torch.tensor(features2, dtype=torch.float32)  # (n2, d)
        self.register_buffer('features1', f1)
        self.register_buffer('features2', f2)

        n1, n2 = f1.shape[0], f2.shape[0]

        # 4. Pre-compute and normalize Feature Distance Matrix — (n1, n2)
        feat_dist = torch.cdist(f1, f2, p=2) ** 2
        self.register_buffer('feat_dist_matrix', feat_dist / (feat_dist.sum() + epsilon))

        # 5. Pre-compute and normalize Index Proximity Matrix — (n1, n2)
        #    Normalize indices to [0, 1] so that sequences of different lengths
        #    are compared on the same relative scale
        idx1 = torch.arange(n1, dtype=torch.float32) / (n1-1)  # (n1,)
        idx2 = torch.arange(n2, dtype=torch.float32) / (n2-1)  # (n2,)
        index_dist = (idx1.unsqueeze(1) - idx2.unsqueeze(0)) ** 2  # (n1, n2)
        self.register_buffer('index_dist', index_dist / (index_dist.sum() + epsilon))

    def get_constrained_sigma(self):
        # sigma is (n1, n2): each row (feature in seq1) sums to 1
        sigma = torch.exp(self.sigma_logits) * (self.sigma_logits > -4)
        sigma = sigma / (sigma.sum(dim=1, keepdim=True) + 1e-9)
        return sigma

    def compute_features_distance(self, sigma):
        return torch.sum(sigma * self.feat_dist_matrix) / sigma.shape[0]

    def compute_index_proximity_cost(self, sigma):
        return torch.sum(sigma * self.index_dist) / sigma.shape[0]

    def forward(self):
        sigma = self.get_constrained_sigma()

        f_dist = self.compute_features_distance(sigma)
        idx_dist = self.compute_index_proximity_cost(sigma)

        return f_dist + self.alpha * idx_dist


def train_model(model):
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

    # Train the model
    model.train()
    for epoch in range(EPOCHS):
        # Calculate the loss
        match_loss = model()
        # Backpropagation
        optimizer.zero_grad()
        match_loss.backward()
        optimizer.step()

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
                scheduler.step()
                lr *= 0.1
                epochs_without_improvement = 0

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_loss
