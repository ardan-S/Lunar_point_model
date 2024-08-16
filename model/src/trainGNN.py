import torch
import torch.optim as optim
import torch_geometric.data as geo_data
# import torch_geometric.loader as geo_loader
# from torch_geometric.utils import subgraph
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import time
import argparse
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import sys

from models import GCN
from custom_loss import FocalLoss, HuberLoss
from utils import load_data, create_GCN_loader, gen_rand_data, stratified_split_data, plot_metrics, change_grid_resolution
from evaluate import evaluate, validate


"""
Considerations for training techniques:
Early Stopping: Monitor validation performance and stop training when performance stops improving to prevent overfitting.
Cross-Validation: Use k-fold cross-validation to ensure the model's performance is consistent across different subsets of the data.
Batch Normalization and Layer Normalization: Experiment with different normalization techniques to stabilize and accelerate training.
"""


def prepare_data(coords, features, labels, k, normalise_scalar=None, standardise_scalar=None):
    # Apply scalars if they are not None
    if standardise_scalar is not None and normalise_scalar is not None:
        features = normalise_scalar.transform(standardise_scalar.transform(features))
    else:
        standardise_scalar = StandardScaler()
        normalise_scalar = MinMaxScaler()
        features = normalise_scalar.fit_transform(standardise_scalar.fit_transform(features))

    x = torch.tensor(np.hstack((coords, features)), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)

    batch_size = 2000  #    OPTIMISE !!!!!!!!!!!!!!!!!!!!
    num_batches = int(np.ceil(len(coords) / batch_size))

    indices = np.zeros((len(coords), k), dtype=int)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(coords))
        _, batch_indices = nbrs.kneighbors(coords[start_idx:end_idx])   # Distances are unused 
        indices[start_idx:end_idx] = batch_indices
        if num_batches >= 10 and i % (num_batches // 10) == 0:
            torch.cuda.empty_cache()
            del batch_indices

    indices = np.vstack(indices)

    num_edges = len(coords) * (k - 1)

    edge_index = torch.empty((2, num_edges), dtype=torch.long)
    chunk_size = 10_000
    idx = 0
    num_nodes = x.size(0)
    num_chunks = int(np.ceil(len(coords) / chunk_size))

    for chunk_start in range(0, len(coords), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(coords))
        _, batch_indices = nbrs.kneighbors(coords[chunk_start:chunk_end])

        batch_indices = torch.tensor(batch_indices, dtype=torch.long)   # Shape: (batch_size, k) - contains the k nearest neighbours for each node in the batch

        for i in range(batch_indices.shape[0]):
            for j in range(1, k):
                node_idx = chunk_start + i
                neighbor_idx = batch_indices[i, j]

                # Additional check to ensure indices are within bounds
                if node_idx >= num_nodes or neighbor_idx >= num_nodes:
                    raise ValueError(f"Invalid index detected: node_idx ({node_idx}) or neighbor_idx ({neighbor_idx}) is out of bounds with num_nodes ({num_nodes}).")

                edge_index[0, idx] = node_idx
                edge_index[1, idx] = neighbor_idx
                idx += 1

        del batch_indices
        torch.cuda.empty_cache()

    # Trim the tensor to the actual number of edges, if necessary
    edge_index = edge_index[:, :idx].clone().detach().t().contiguous()

    # Check for any indices out of bounds
    num_nodes = x.size(0)
    if edge_index.max().item() >= num_nodes or edge_index.min().item() < 0:
        print("ERROR: edge_index contains invalid node indices.")
        print(f"indices.max ({edge_index.max()}) >= num_nodes ({num_nodes}) or indices.min ({edge_index.min()}) < 0")
        raise ValueError("fix it")

    """
    Comparing fully connected GCN with K nearest neighbours GCN:
    
    Fully connected GCN:
    Each node recieves information from all other nodes, this can capture global dependencies. 
    High computational complexity O(n^2) where n is the number of nodes.
    High redundancy if not all connections carry meaningful information.
    Given the memory issues with knn, this was discounted as not feasible

    K nearest neighbours GCN:
    Each node recieves information from a fixed number of neighbours, this can capture local dependencies.
    Lower computational complexity O(nk) where k is the number of neighbours.
    Less redundancy as only meaningful connections are considered.
    Better when local interactions are more meaningful than global ones
    """

    return geo_data.Data(x=x, edge_index=edge_index, y=y)


def setup_GCN_data(args):
    rand_state = 42
    torch.manual_seed(rand_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example data
    # npoints = 1_000
    # labeled_data = gen_rand_data(npoints, rand_state) 

    # Load data
    labeled_data = load_data(args.data_path)

    labeled_data = labeled_data[labeled_data["Latitude"] < 0]   # Only train on southern hemisphere and really close to the pole
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.0002, random_state=rand_state)
    _, test_index = next(sss.split(labeled_data, labeled_data["Label"]))
    labeled_data = labeled_data.iloc[test_index]
    print(f"Size of selected dataset: {labeled_data.shape[0]}")

    labeled_x = labeled_data[['Latitude', 'Longitude', 'Diviner', 'LOLA', 'M3', 'MiniRF']]
    labeled_y = labeled_data['Label']

    input_dim = labeled_x.shape[1]

    train_x, val_x, test_x, train_y, val_y, test_y = stratified_split_data(labeled_x, labeled_y, rand_state=rand_state)

    def split_data(data_x, data_y):
        coords = data_x[['Latitude', 'Longitude']]
        features = data_x[['Diviner', 'LOLA', 'M3', 'MiniRF']]
        labels = data_y.values
        return coords, features, labels
    
    train_coords, train_features, train_targets = split_data(train_x, train_y)
    val_coords, val_features, val_targets = split_data(val_x, val_y)
    test_coords, test_features, test_targets = split_data(test_x, test_y)

    standardise_scalar = StandardScaler().fit(train_features)
    normalise_scalar = MinMaxScaler().fit(standardise_scalar.transform(train_features))

    train_graph_data = prepare_data(train_coords, train_features, train_targets, args.k, normalise_scalar, standardise_scalar)
    val_graph_data = prepare_data(val_coords, val_features, val_targets, args.k, normalise_scalar, standardise_scalar)
    test_graph_data = prepare_data(test_coords, test_features, test_targets, args.k, normalise_scalar, standardise_scalar)

    return device, input_dim, train_graph_data, val_graph_data, test_graph_data


def setup_GCN_loader(train_graph_data, val_graph_data, test_graph_data, device, args):
    train_loader = create_GCN_loader([train_graph_data], device, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = create_GCN_loader([test_graph_data], device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    val_loader = create_GCN_loader([val_graph_data], device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    return train_loader, val_loader, test_loader


def setup_GCN_model(input_dim, args, device):
    output_dim = 1
    model = GCN(input_dim, args.hidden_dim, output_dim, args.dropout_rate).to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler()    # Initialise GradScaler for mixed precision training

    return model, criterion, optimiser, scaler


def train_GCN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, model_save_path=None, img_save_path=None):
    start_time = time.time()
    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []

    print(f"\nEntering training loop after {(time.time() - start_time) / 60 :.2f} mins")
    sys.stdout.flush()
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimiser.zero_grad()

            # Use autocast to enable mixed precision training
            with torch.amp.autocast('cuda'):

                if data.edge_index.max() >= data.x.size(0) or data.edge_index.min() < 0:
                    print(f"Invalid edge_index. Max: {data.edge_index.max()}, Min: {data.edge_index.min()}")
                    print(f"Max index: {data.x.size(0) - 1}")
                    raise ValueError("edge_index contains invalid node indices.")
                
                max_index = data.x.size(0) - 1
                if not torch.all((data.edge_index[0] <= max_index) & (data.edge_index[1] <= max_index)):
                    print(f"Edge indices out of bounds. Max index: {max_index}")
                    raise ValueError("edge_index out of bounds in edge_index tensor.")
                
                data.edge_index = data.edge_index.t()

                outputs = model(data.x, data.edge_index).squeeze()
                loss = criterion(outputs, data.y.float())

            # loss.backward()
            # optimiser.step()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item() * data.num_graphs

        epoch_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_mse, val_r2 = validate(device, model, criterion, val_loader)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_r2s.append(val_r2)

        print(f'Epoch [{epoch+1:02d}/{args.num_epochs}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Val mse: {val_mse:.4f}, Val R²: {val_r2:.4f}, Time: {time.time() - epoch_start:.2f}s')
        sys.stdout.flush()

    print(f"Training completed in {(time.time() - start_time)/60 :.2f} mins")
    test_mse, test_mae, test_r2 = evaluate(device, model, test_loader)

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    if img_save_path:
        plot_metrics(args.num_epochs, train_losses, val_losses, val_mses, val_r2s, test_mse, test_mae, test_r2, save_path=img_save_path)

    return model, test_mse, test_mae, test_r2

def main():
    args = parse_arguments()
    start_time = time.time()
    device, input_dim, train_graph_data, val_graph_data, test_graph_data = setup_GCN_data(args)
    print(f"Data preparation completed in {(time.time() - start_time) / 60 :.2f} mins")
    train_loader, val_loader, test_loader = setup_GCN_loader(train_graph_data, val_graph_data, test_graph_data, device, args)
    print(f"Data loaders setup completed in {(time.time() - start_time) / 60 :.2f} mins")
    model, criterion, optimiser, scaler = setup_GCN_model(input_dim, args, device)
    print(f"Model setup completed in {(time.time() - start_time) / 60 :.2f} mins")
    model, test_mse, test_mae, test_r2 = train_GCN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, img_save_path='../figs/training_metrics_GCN.png')
    print(f"Training completed in {(time.time() - start_time) / 60 :.2f} mins")
    print("\nTest set:")
    print(f'Mean Squared Error (MSE): {test_mse:.4f}')
    print(f'Mean Absolute Error (MAE): {test_mae:.4f}')
    print(f'R-squared (R²): {test_r2:.4f}\n')
    

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