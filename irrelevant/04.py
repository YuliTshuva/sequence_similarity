"""
Yuli Tshuva
Training an RNN for feature extraction of a segment.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# 1. The LSTM Encoder
class SequenceEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, embedding_dim=32):
        super(SequenceEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Because it's bidirectional, we get hidden_dim from Forward AND hidden_dim from Backward
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)

        # h_n has shape (num_layers * num_directions, batch, hidden_dim)
        # We concatenate the last forward state and the last backward state
        last_forward = h_n[-2, :, :]
        last_backward = h_n[-1, :, :]
        combined = torch.cat((last_forward, last_backward), dim=1)

        return self.fc(combined)


# 2. Setup Hyperparameters
input_dim = 1
hidden_dim = 128
embedding_dim = 64
learning_rate = 1e-3
margin = 1.0

model = SequenceEncoder(input_dim, hidden_dim, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.TripletMarginLoss(margin=margin, p=2)


# 3. Dummy Training Loop Logic
# Note: anchor, positive, and negative must all be (batch, seq_len, 1)
def train_step(anchor, positive, negative):
    model.train()
    optimizer.zero_grad()

    # Pass all three through the SAME encoder
    anchor_embed = model(anchor)
    pos_embed = model(positive)
    neg_embed = model(negative)

    # Calculate Triplet Loss
    loss = criterion(anchor_embed, pos_embed, neg_embed)

    loss.backward()
    optimizer.step()
    return loss.item()


# --- Example Usage with Dummy Data ---
# batch_size=8, seq_len=20, features=1
anc_data = torch.randn(8, 20, 1)
pos_data = torch.randn(8, 20, 1)  # In reality, this should be 'similar' to anc
neg_data = torch.randn(8, 20, 1)  # In reality, this should be 'dissimilar' to anc

loss_val = train_step(anc_data, pos_data, neg_data)
print(f"Current Loss: {loss_val:.4f}")