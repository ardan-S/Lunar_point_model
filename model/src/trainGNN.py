import torch
import torch.optim as optim
import torch_geometric.data as geo_data
import torch_geometric.loader as geo_loader
from torch_geometric.utils import subgraph
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import time
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

from models import GCN
from custom_loss import FocalLoss, HuberLoss
from utils import create_GCN_loader, gen_rand_data, stratified_split_data, plot_metrics
from evaluate import evaluate, validate


"""
Considerations for training techniques:
Early Stopping: Monitor validation performance and stop training when performance stops improving to prevent overfitting.
Cross-Validation: Use k-fold cross-validation to ensure the model's performance is consistent across different subsets of the data.
Batch Normalization and Layer Normalization: Experiment with different normalization techniques to stabilize and accelerate training.
"""

def load_data(data_path):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Directory {data_path} not found")
    
    all_csvs = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    filepaths = [os.path.join(data_path, f) for f in all_csvs]

    data = pd.concat([pd.read_csv(file) for file in filepaths], ignore_index=True)

    return data

def prepare_data(labeled_data, k):
    global start_time
    coords = labeled_data[['Latitude', 'Longitude']].values  # Shape: (num_nodes, 2)
    features = labeled_data[['Diviner', 'LOLA', 'M3', 'MiniRF']].values  # Shape: (num_nodes, 4)
    labels = labeled_data['Label'].values  # Shape: (num_nodes,)
    print(f"Num unique labels: {len(np.unique(labels))}")
    print(f"Labels before conversion: {np.unique(labels)}")
    sys.stdout.flush()
    
    standardise_scalar = StandardScaler()
    normalise_scalar = MinMaxScaler()
    features = normalise_scalar.fit_transform(standardise_scalar.fit_transform(features))
    print("Standardised and normalised features")
    sys.stdout.flush()

    # Combine coordinates and features into a single tensor
    x = torch.tensor(np.hstack((coords, features)), dtype=torch.float)  # Shape: (num_nodes, 6)
    y = torch.tensor(labels, dtype=torch.float)  # Shape: (num_nodes,)

    print(f"Num inque labels after conversion: {len(torch.unique(y))}")
    print(f"Labels after conversion: {torch.unique(y)}. Completed after {time.time() - start_time :.2f} seconds")
    sys.stdout.flush()
    
    # Fully connected graph
    # num_nodes = x.size(0)
    # edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()

    # K nearest neighbours graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    # distances, indices = nbrs.kneighbors(coords)

    batch_size = 1000  #    OPTIMISE !!!!!!!!!!!!!!!!!!!!
    num_batches = int(np.ceil(len(coords) / batch_size))

    distances = []
    indices = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(coords))
        batch_distances, batch_indices = nbrs.kneighbors(coords[start_idx:end_idx])
        distances.append(batch_distances)
        indices.append(batch_indices)
        print(f"Batch {i+1}/{num_batches} completed after {time.time() - start_time :.2f} seconds")
        sys.stdout.flush()

    distances = np.vstack(distances)
    indices = np.vstack(indices)

    print("Nearest neighbours calculated")
    sys.stdout.flush()

    edge_index = []
    for i in range(len(indices)):
        for j in range(1, k):
            edge_index.append([i, indices[i, j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    print(f"x shape: {x.shape}")
    print(f"edge index shape: {edge_index.shape}")
    print(f"edge_index max value: {edge_index.max()}")
    print(f"edge_index min value: {edge_index.min()}")
    print(f"y shape: {y.shape}")
    print(f"y unique values: {torch.unique(y)}")
    sys.stdout.flush()

    # Check for any indices out of bounds
    num_nodes = x.size(0)
    if edge_index.max() >= num_nodes or edge_index.min() < 0:
        raise ValueError("edge_index contains invalid node indices.")

    """
    Comparing fully connected GCN with K nearest neighbours GCN:
    
    Fully connected GCN:
    Each node recieves information from all other nodes, this can capture global dependencies. 
    High computational complexity O(n^2) where n is the number of nodes.
    High redundancy if not all connections carry meaningful information.

    K nearest neighbours GCN:
    Each node recieves information from a fixed number of neighbours, this can capture local dependencies.
    Lower computational complexity O(nk) where k is the number of neighbours.
    Less redundancy as only meaningful connections are considered.
    Better when local interactions are more meaningful than global ones
    """

    return geo_data.Data(x=x, edge_index=edge_index, y=y)


def train(args):
    global start_time
    start_time = time.time()

    rand_state = 42
    torch.manual_seed(rand_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = 1 

    # Example data
    # npoints = 1_000
    # labeled_data = gen_rand_data(npoints, rand_state) 

    print("Loading data...")
    sys.stdout.flush()
    # Load data
    labeled_data = load_data(args.data_path)

    print("Preparing data...")
    sys.stdout.flush()
    graph_data = prepare_data(labeled_data, args.k)

    input_dim = graph_data.x.size(1)
    print(f"Input dimension: {input_dim}")
    sys.stdout.flush()

    train_features, val_features, test_features, train_targets, val_targets, test_targets = stratified_split_data(graph_data.x, graph_data.y)
    print(f"Data split after {time.time() - start_time :.2f} seconds")
    sys.stdout.flush()

    def create_subgraph(features, targets, edge_index):
        subset = torch.arange(features.size(0))
        edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
        return geo_data.Data(x=features, edge_index=edge_index, y=targets)

    train_data = create_subgraph(train_features, train_targets, graph_data.edge_index)
    val_data = create_subgraph(val_features, val_targets, graph_data.edge_index)
    test_data = create_subgraph(test_features, test_targets, graph_data.edge_index)
    print(f"Subgraphs created after {time.time() - start_time :.2f} seconds")
    sys.stdout.flush()

    train_loader = create_GCN_loader([train_data], device, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = create_GCN_loader([test_data], device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    val_loader = create_GCN_loader([val_data], device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print(f"Data loaders created after {time.time() - start_time :.2f} seconds")
    sys.stdout.flush()

    model = GCN(input_dim, args.hidden_dim, output_dim, args.dropout_rate)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    # criterion = FocalLoss(alpha=1, gamma=2)
    criterion = HuberLoss(delta=1.0)

    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []

    print(f"Entering training loop after {time.time() - start_time :.2f} seconds")
    sys.stdout.flush()
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimiser.zero_grad()
            print("1")


            if data.edge_index.max() >= data.x.size(0) or data.edge_index.min() < 0:
                print(f"Invalid edge_index. Max: {data.edge_index.max()}, Min: {data.edge_index.min()}")
                print(f"Max index: {data.x.size(0) - 1}")
                raise ValueError("edge_index contains invalid node indices.")
            
            max_index = data.x.size(0) - 1
            if not torch.all((data.edge_index[0] <= max_index) & (data.edge_index[1] <= max_index)):
                print(f"Edge indices out of bounds. Max index: {max_index}")
                raise ValueError("edge_index out of bounds in edge_index tensor.")

            outputs = model(data.x, data.edge_index).squeeze()

            print("2")
            loss = criterion(outputs, data.y.float())
            running_loss += loss.item() * data.num_graphs

            loss.backward()
            optimiser.step()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_mse, val_r2 = validate(device, model, criterion, test_loader)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_r2s.append(val_r2)

        print(f'Epoch [{epoch+1:02d}/{args.num_epochs}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Val mse: {val_mse:.4f}, Val R²: {val_r2:.4f}, Time: {time.time() - epoch_start:.2f}s')
        sys.stdout.flush()

    print(f"Training completed in {time.time() - start_time :.2f} seconds")

    # # Save the model for evaluation
    # torch.save(model.state_dict(), 'point_ranking_model.pth')

    # Evaluate the model on the test set
    test_mse, test_mae, test_r2 = evaluate(device, model, test_loader)
    print("\nTest set:")
    print(f'Mean Squared Error (MSE): {test_mse:.4f}')
    print(f'Mean Absolute Error (MAE): {test_mae:.4f}')
    print(f'R-squared (R²): {test_r2:.4f}\n')

    # Plot the training and validation losses
    plot_metrics(args.num_epochs, train_losses, val_losses, val_mses, val_r2s, test_mse, test_mae, test_r2, save_path='../figs/training_metrics_GCN.png')

def main():
    args = parse_arguments()
    train(args)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a PointRankingModel.')
    parser.add_argument('--data_path', type=str, default='../../data/Combined_CSVs', help='Path to the input data file.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the hidden layer.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders.')
    parser.add_argument('--k', type=int, default=100, help='Number of nearest neighbours for KNN graph')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for data loaders.')
    return parser.parse_args()

if __name__ == '__main__':
    main()