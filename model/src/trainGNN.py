import torch
import torch.optim as optim
import torch_geometric.data as geo_data
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

import time
import argparse
import numpy as np

from models import GCN
from utils import load_data, create_GCN_loader, stratified_split_data, plot_metrics, balanced_sample
from evaluate import evaluate, validate


def prepare_data(coords, features, labels, k, normalise_scalar=None, standardise_scalar=None, scale_labels=False):

    if standardise_scalar is not None:
        features = standardise_scalar.transform(features)
        if scale_labels:
            labels = standardise_scalar.transform(labels.reshape(-1, 1)).reshape(-1)

    if normalise_scalar is not None:
        features = normalise_scalar.transform(features)
        if scale_labels:
            labels = normalise_scalar.transform(labels.reshape(-1, 1)).reshape(-1)

    x = torch.tensor(np.hstack((coords, features)), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)

    batch_size = 2000 
    num_batches = int(np.ceil(len(coords) / batch_size))

    indices = np.zeros((len(coords), k), dtype=int)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(coords))
        _, batch_indices = nbrs.kneighbors(coords[start_idx:end_idx])
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

    if edge_index.max().item() >= x.size(0) or edge_index.min().item() < 0:
        raise ValueError("Invalid edge_index detected: edge_index contains out-of-bounds node indices.")

    return geo_data.Data(x=x, edge_index=edge_index, y=y)


def setup_GCN_data(args):
    rand_state = 42
    torch.manual_seed(rand_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_data = load_data(args.data_path)
    labeled_data = balanced_sample(labeled_data, 'Label', 0.001, random_state=rand_state)

    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.0025, random_state=rand_state)
    # _, test_index = next(sss.split(labeled_data, labeled_data["Label"]))
    # labeled_data = labeled_data.iloc[test_index]
    # print(f"Size of selected dataset: {labeled_data.shape[0]}")

    labeled_x = labeled_data[['Latitude', 'Longitude', 'Diviner', 'LOLA', 'M3', 'MiniRF', 'Elevation']]
    labeled_y = labeled_data['Label']
    
    input_dim = labeled_x.shape[1]

    train_x, val_x, test_x, train_y, val_y, test_y = stratified_split_data(labeled_x, labeled_y, rand_state=rand_state)

    def split_data(data_x, data_y):
        coords = data_x[['Latitude', 'Longitude']]
        features = data_x[['Diviner', 'LOLA', 'M3', 'MiniRF', 'Elevation']]
        labels = data_y.values
        return coords, features, labels
    
    train_coords, train_features, train_targets = split_data(train_x, train_y)
    val_coords, val_features, val_targets = split_data(val_x, val_y)
    test_coords, test_features, test_targets = split_data(test_x, test_y)

    standardise_scalar = StandardScaler().fit(train_features)
    normalise_scalar = MinMaxScaler().fit(standardise_scalar.transform(train_features))
    joblib.dump(standardise_scalar, '../saved_models/standardise_scalar_GCN.joblib')
    joblib.dump(normalise_scalar, '../saved_models/normalise_scalar_GCN.joblib')

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
    model = GCN(input_dim, args.hidden_dim, 1, args.dropout_rate).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=args.beta)
    scaler = torch.amp.GradScaler()    # Initialise GradScaler for mixed precision training

    return model, criterion, optimiser, scaler


def compute_gradient_norms(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Compute L2 norm of the gradient
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def train_GCN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, model_save_path=None, img_save_path=None):
    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []

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

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            # grad_norm = compute_gradient_norms(model)
            # print(f'Epoch {epoch+1}, Gradient Norm: {grad_norm}')

            running_loss += loss.item() * data.num_graphs

        epoch_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_mse, val_r2 = validate(device, model, criterion, val_loader)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_r2s.append(val_r2)

        print(f'Epoch [{epoch+1:02d}/{args.num_epochs}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Val mse: {val_mse:.4f}, Val R²: {val_r2:.4f}, Time: {time.time() - epoch_start:.2f}s')

    test_loss, test_mse, test_r2 = evaluate(device, model, criterion, test_loader)

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        file_path = model_save_path.replace('.pth', '.txt')
        content = f"""
        Parameters of saved GCN:
        Hidden dimension: {args.hidden_dim}
        Epochs: {args.num_epochs}
        Learning rate: {args.learning_rate}
        Dropout rate: {args.dropout_rate}
        Batch size: {args.batch_size}
        K: {args.k}
        Beta: {args.beta}
        Weight decay: {args.weight_decay}

        Test set:
        Mean Squared Error (MSE): {test_mse:.4f}
        Loss: {test_loss:.4f}
        R-squared (R²): {test_r2:.4f}

        Saved on {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Model saved at {model_save_path}")

    if img_save_path:
        plot_metrics(args.num_epochs, train_losses, val_losses, test_loss, val_mses, val_r2s, test_mse, test_r2, save_path=img_save_path)

    return model, test_loss, test_mse, test_r2

def main():
    args = parse_arguments()
    start_time = time.time()
    device, input_dim, train_graph_data, val_graph_data, test_graph_data = setup_GCN_data(args)
    print(f"Data preparation completed in {(time.time() - start_time) / 60 :.2f} mins")
    train_loader, val_loader, test_loader = setup_GCN_loader(train_graph_data, val_graph_data, test_graph_data, device, args)
    print(f"Data loaders setup completed in {(time.time() - start_time) / 60 :.2f} mins")
    model, criterion, optimiser, scaler = setup_GCN_model(input_dim, args, device)
    print(f"Model setup completed in {(time.time() - start_time) / 60 :.2f} mins")
    model, test_loss, test_mse, test_r2 = train_GCN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, img_save_path='../figs/training_metrics_GCN.png', model_save_path='../saved_models/GCN.pth')
    print(f"Training completed in {(time.time() - start_time) / 60 :.2f} mins")
    print("\nTest set:")
    print(f'Mean Squared Error (MSE): {test_mse:.4f}')
    print(f'Loss: {test_loss:.4f}')
    print(f'R-squared (R²): {test_r2:.4f}\n')

    # Get random graph from test loader
    random_graph = next(iter(test_loader))
    inputs, targets = random_graph.x, random_graph.y
    inputs, targets = inputs.to(device), targets.to(device)

    edge_index = random_graph.edge_index
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    
    edge_index, _ = add_self_loops(edge_index, num_nodes=inputs.size(0))
    edge_index = edge_index.to(device)
    outputs = model(inputs, edge_index).squeeze()

    targets_np = targets.cpu().detach().numpy()[:5]
    outputs_np = outputs.cpu().detach().numpy()[:5]

    print(f"True labels (first 4): {targets_np}")
    print(f"Model output (first 4): {outputs_np}")

    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a PointRankingModel.')
    parser.add_argument('--data_path', type=str, default='../../data/Combined_CSVs', help='Path to the input data file.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the hidden layer.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders.')
    parser.add_argument('--k', type=int, default=100, help='Number of nearest neighbours for KNN graph')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta for the smooth L1 loss.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for the optimizer.')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for data loaders.')
    return parser.parse_args()

if __name__ == '__main__':
    main()