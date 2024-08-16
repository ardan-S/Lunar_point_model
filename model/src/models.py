import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import time
import sys

"""
Key
fc: Fully connected layer
bn: Batch normalization
dropout: Dropout layer
"""

"""Attention layer:
This allows the model to focus on important features, potentially improving its
ability to learn relevant patterns in the data, particularly if some features 
(e.g., specific remote sensing values) are more indicative of water presence.
The model can learn to focus on these important features dynamically."""

"""Residual connection:
Improves gradient flow, enabling the model to learn more effectively even with
deeper networks. This can capture more complex patterns in the data."""


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attention_weights = self.softmax(scores)
        weighted_sum = torch.matmul(attention_weights, v)
        return weighted_sum


class FCNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=1, dropout_rate=0.3):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.self_attention = SelfAttentionLayer(hidden_dim)
        self.residual_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        residual = x
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)) + self.residual_fc(residual))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = self.self_attention(x.unsqueeze(1))
        x = self.output(x.squeeze(1))
        return x


    """The initial FC layers are effective at extracting and transforming raw input features into meaningful high-dimensional representations.
    The self-attention mechanism can then effectively capture dependencies and interactions between these high-dimensional features.
    This structure allows the model to leverage the strengths of both fully connected layers for feature extraction and 
        attention mechanisms for dependency modeling, leading to more accurate and robust predictions."""
    

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.3):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=4)
        self.residual_conv = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)

        residual = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)) + self.residual_conv(residual))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = self.dropout(x)
        
        print("Not using attention layer")
        # x = x.unsqueeze(1)  # Convert to (batch_size, seq_len, feature_dim) format
        # x, _ = self.attention(x, x, x)
        # x = x.squeeze(1)

        # COULD CONSIDER- Global mean pooling to aggregate node features into graph-level features
        x = self.output(x)
        return x

"""
Self-Attention Layer: Similar to the FCNN, the GCN incorporates a self-attention mechanism (in this case, using MultiheadAttention from PyTorch)
Residual Connections: The GCN also includes residual connections between layers, which help to prevent vanishing gradients and allow the model to learn more complex relationships.
Batch Normalization: Like the FCNN, batch normalization is used in each layer to stabilize training and improve convergence.
Global Mean Pooling: After applying the GCN and attention layers, a global mean pooling layer aggregates the node-level information into a graph-level feature vector, which is then used for regression.
"""