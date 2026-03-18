"""
Yuli Tshuva
Create a model for sequence similarity based on the graph distance algorithm.
"""

# Imports
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from tqdm.auto import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA, BETA = 1, 1e10
LR, MIN_LR = 1e-3, 1e-8
FORCE_REGULARIZATION_FREQUENCY = 4
PATIENCE = 10
EPOCHS = 2000


class SequenceSimilarity(nn.Module):
    def __init__(self, initial_match_matrix, features1, features2):
        """
        param: initial_match_matrix: A 2D numpy array representing the initial similarity between nodes
               of the two sequences, based on their features.
        features1: A 2D numpy array of shape (n_nodes_seq1, feature_dim) representing the features of nodes in sequence 1.
        features2: A 2D numpy array of shape (n_nodes_seq2, feature_dim) representing the features of nodes in sequence 2.
        """
        super(SequenceSimilarity, self).__init__()
        # Process the initial match matrix to create a learnable parameter
        self.sigma = torch.tensor(initial_match_matrix, dtype=torch.float32, device=DEVICE)
        self.sigma = Parameter(self.sigma, requires_grad=True)
        # Set the features of each sequence
        self.features1 = torch.tensor(features1, dtype=torch.float32, device=DEVICE)
        self.features2 = torch.tensor(features2, dtype=torch.float32, device=DEVICE)
        # Find the sum of each row and column of the initial match matrix
        self.row_sums = torch.sum(self.sigma, dim=1)
        self.col_sums = torch.sum(self.sigma, dim=0)

    def compute_features_distance(self):
        # Calculate a matrix where the (i, j) entry is the distance between features of node i in sequence 1 and node j in sequence 2
        # Using squared Euclidean distance
        features1_expanded = self.features1.unsqueeze(1)  # Shape: (n, 1, feature_dim)
        features2_expanded = self.features2.unsqueeze(0)  # Shape: (1, m, feature_dim)
        distance_matrix = torch.sum((features1_expanded - features2_expanded) ** 2, dim=2)  # Shape: (n, m)
        # Multiply the distance matrix by the learnable sigma to get a weighted distance
        weighted_distance = self.sigma * distance_matrix
        # Sum over all pairs to get a single similarity score (or distance)
        similarity_score = torch.sum(torch.sum(weighted_distance))
        return similarity_score

    @staticmethod
    def compute_pairwise_structure_distance(i, j, k, l):
        temporal_distance = (abs(i - j) - abs(k - l)) ** 2
        return temporal_distance

    def compute_structure_distance(self):
        # Set a variable to store the total structure distance error
        total_error = torch.tensor([0], device=DEVICE, dtype=torch.float32)
        # Find the number of nodes in each sequence
        n1, n2 = self.features1.shape[0], self.features2.shape[0]
        # Iterate over all pairs of nodes in both sequences and calculate the structure distance error based on the current mapping (sigma)
        for i, j in zip(range(n1), range(n1)):
            for k, l in zip(range(n2), range(n2)):
                total_error += self.sigma[i, k] * self.sigma[j, l] * self.compute_pairwise_structure_distance(i, j, k,
                                                                                                              l)
        return total_error

    def forward(self):
        return self.compute_features_distance() + ALPHA * self.compute_structure_distance()

    def regularization_loss(self):
        """Add a regularization term to prevent weights collapse"""
        rows_deviation = torch.sum((torch.sum(self.sigma, dim=1) - self.row_sums) ** 2)
        cols_deviation = torch.sum((torch.sum(self.sigma, dim=0) - self.col_sums) ** 2)
        return rows_deviation + cols_deviation


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
    for epoch in tqdm(range(EPOCHS), desc="Training", total=EPOCHS):
        # Calculate the loss
        match_loss = model()
        if epoch % FORCE_REGULARIZATION_FREQUENCY == 0:
            reg_loss = model.regularization_loss()
            loss = match_loss + BETA * reg_loss
        else:
            loss = match_loss
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check for improvement
        if match_loss.item() < best_loss:
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
