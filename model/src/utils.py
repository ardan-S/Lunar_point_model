import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import resample
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader  # Avoid name conflict with PyTorch DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_processing.label import apply_labels


def load_data(data_path, output=True):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Directory {data_path} not found")
    
    all_csvs = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    filepaths = [os.path.join(data_path, f) for f in all_csvs]

    data = pd.concat([pd.read_csv(file) for file in filepaths], ignore_index=True)

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    label_counts = Counter(data['Label'])
    total_count = sum(label_counts.values())
    label_percentages = {label: (count / total_count) * 100 for label, count in label_counts.items()}
    if output:
        print("Percentage of each label in the dataset:")
        for label in sorted(label_percentages.keys()):
            print(f"Label {label}: {label_percentages[label]:.5f}%")
    
    return data


def balanced_sample(data, target_column, fraction, random_state=42):
    total_samples = int(len(data) * fraction)
    label_0_target = total_samples *2 //9
    other_label_target = total_samples // 9
    
    grouped = data.groupby(target_column)
    
    samples = []
    sampled_counts = {i: 0 for i in range(8)}
    
    # First, try to sample the desired number of examples from each label
    for label in range(8):
        if label in grouped.groups:
            label_data = grouped.get_group(label)
            target_samples = label_0_target if label == 0 else other_label_target
            available_samples = len(label_data)
            
            if available_samples >= target_samples:
                sampled_data = resample(label_data, replace=False, n_samples=target_samples, random_state=random_state)
                sampled_counts[label] += target_samples
            else:
                sampled_data = label_data
                sampled_counts[label] += available_samples
                
            samples.append(sampled_data)
    
    current_total_samples = sum(sampled_counts.values())
    remaining_samples = total_samples - current_total_samples
    
    # Keep sampling from the smallest available datasets until we reach the target size
    while remaining_samples > 0:
        for label in range(8):
            if label in grouped.groups and remaining_samples > 0:
                label_data = grouped.get_group(label)
                target_samples = min(len(label_data), remaining_samples)
                
                if sampled_counts[label] < len(label_data):
                    additional_samples = min(target_samples, len(label_data) - sampled_counts[label])
                    sampled_data = resample(label_data, replace=False, n_samples=additional_samples, random_state=random_state)
                    samples.append(sampled_data)
                    sampled_counts[label] += additional_samples
                    remaining_samples -= additional_samples
    
    balanced_data = pd.concat(samples).reset_index(drop=True)
    final_proportions = {label: count / total_samples for label, count in sampled_counts.items()}

    print("Percentage of each label after resampling:")
    for label, proportion in final_proportions.items():
        print(f"Label {label}: {proportion*100:.2f}%")
    
    return balanced_data


def create_FCNN_loader(inputs, targets, device, batch_size=128, shuffle=True, num_workers=1, standardise_scalar=None, normalise_scalar=None, scale_targets=False):
    """
    Create a DataLoader for the given dataset.

    Args:
    - inputs: Tensor of input features.
    - targets: Tensor of input labels.
    - batch_size: Size of the batches.
    - shuffle: Whether to shuffle the data.

    Returns:
    - DataLoader for the dataset.
    """

    if standardise_scalar is not None:
        if not isinstance(inputs, pd.DataFrame):
            inputs = pd.DataFrame(inputs.numpy()) if torch.is_tensor(inputs) else pd.DataFrame(inputs)
        inputs = standardise_scalar.transform(inputs)
        inputs = pd.DataFrame(inputs, columns=standardise_scalar.feature_names_in_)
    else:
        raise ValueError("Standardisation scalar must be provided for FCNN")

    if normalise_scalar is not None:
        if not isinstance(inputs, pd.DataFrame):
            inputs = pd.DataFrame(inputs.numpy()) if torch.is_tensor(inputs) else pd.DataFrame(inputs)
        inputs = normalise_scalar.transform(inputs)
        inputs = pd.DataFrame(inputs, columns=normalise_scalar.feature_names_in_)
    else:
        print("FCNN not using normalisation scalar")

    # Targets not scaled or normalised

    # Convert inputs and targets to tensors
    inputs = torch.tensor(inputs.values, dtype=torch.float32) if not torch.is_tensor(inputs) else inputs
    targets = torch.tensor(targets.values, dtype=torch.float32) if not torch.is_tensor(targets) else targets

    # Create dataset and data loader
    dataset = TensorDataset(inputs, targets)
    if device == 'cuda':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=(device.type == 'cuda'), prefetch_factor=2)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=2)

    return data_loader


def create_GCN_loader(data, device, batch_size=32, shuffle=True, num_workers=1):
    """
    Create a DataLoader for the given graph dataset for GCN.

    Args:
    - features: Tensor of input features.
    - targets: Tensor of input labels.
    - edge_index: Tensor of edge indices.
    - batch_size: Size of the batches.
    - shuffle: Whether to shuffle the data.

    Returns:
    - DataLoader for the dataset.
    """

    if device == 'cuda':
        print("Start of data loader")
        data_loader = GeoDataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=(device.type == 'cuda'), prefetch_factor=2)
        print("End of data loader")
    else:
        data_loader = GeoDataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


def gen_rand_data(npoints=100, rand_state=42):
    torch.manual_seed(rand_state)
    print("!!!! TRAINING ON RANDOM DATA !!!!")
    print(f"Generating {npoints} random examples...")
    sys.stdout.flush()
    inputs = torch.randn(npoints, 6)  # 6 features per example
    
    # Feature1: Scale to range 0 to 360
    inputs[:, 0] -= inputs[:, 0].min()  # Shift to positive values
    inputs[:, 0] /= inputs[:, 0].max()  # Normalize to 0-1
    inputs[:, 0] *= 360  # Scale to 0-360

    # Feature2: Map to two ranges -90 to -75 and 75 to 90
    min_val = inputs[:, 1].min()
    max_val = inputs[:, 1].max()
    inputs[0::2, 1] = -90 + (inputs[0::2, 1] - min_val) / (max_val - min_val) * 15
    inputs[1::2, 1] = 75 + (inputs[1::2, 1] - min_val) / (max_val - min_val) * 15

    # Feature3: Scale to range 35 to 350
    inputs[:, 2] -= inputs[:, 2].min()  # Shift to positive values
    inputs[:, 2] /= inputs[:, 2].max()  # Normalize to 0-1
    inputs[:, 2] = inputs[:, 2] * (350 - 35) + 35  # Scale to 35-350

    # Features4-6: Scale to range 0 to 1
    min_vals = inputs[:, 3:].min(dim=0, keepdim=True)[0]
    max_vals = inputs[:, 3:].max(dim=0, keepdim=True)[0]
    inputs[:, 3:] = (inputs[:, 3:] - min_vals) / (max_vals - min_vals)

    # Convert to pandas DataFrame with specified column names
    df = pd.DataFrame(inputs.numpy(), columns=["Longitude", "Latitude", "Diviner", "LOLA", "M3", "MiniRF"])
    df['Label'] = 0

    # Apply labels (assuming apply_labels is a predefined function)
    labeled_data = apply_labels(df)

    # Normalize label to 0-1
    labeled_data['Label'] /= 7

    print("3 largest values in Label column")
    print(labeled_data.nlargest(3, 'Label'))
    print("\n3 smallest values in Label column")
    print(labeled_data.nsmallest(3, 'Label'))
    print("\nDescription of labeled data")
    print(labeled_data.describe())
    print()

    return labeled_data


def stratified_split_data(features, targets, rand_state=42):
    """
    Splits the data into training, validation, and test sets using stratified splits if possible.
    Falls back to regular splits if stratified splitting is not feasible.

    Parameters:
    - features: The input features of the dataset
    - targets: The target labels of the dataset
    - rand_state: The random seed for reproducibility

    Returns:
    - train_features, val_features, test_features: Features for training, validation, and test sets
    - train_targets, val_targets, test_targets: Targets for training, validation, and test sets
    """
    torch.manual_seed(rand_state)

    try: 
        split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rand_state)
        train_index, test_val_index = next(split1.split(features, targets))
        train_features = features.iloc[train_index]
        test_val_features = features.iloc[test_val_index]
        train_targets = targets.iloc[train_index]
        test_val_targets = targets.iloc[test_val_index]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=rand_state)
        test_index, val_index = next(split2.split(test_val_features, test_val_targets))
        test_features = test_val_features.iloc[test_index]
        val_features = test_val_features.iloc[val_index]
        test_targets = test_val_targets.iloc[test_index]
        val_targets = test_val_targets.iloc[val_index]

    except ValueError:
        print("Stratified splitting failed. Falling back to regular splitting.")
            # Split data into training and validation sets using regular splits
        train_features, test_val_features, train_targets, test_val_targets = train_test_split(
            features, targets, test_size=0.2, random_state=rand_state)
        test_features, val_features, test_targets, val_targets = train_test_split(
            test_val_features, test_val_targets, test_size=0.5, random_state=rand_state)

    # train_features = np.array(train_features)


    return train_features, val_features, test_features, train_targets, val_targets, test_targets


def get_random_filtered_graph(file_path, label_value=None, k=5):
    data = load_data(file_path)

    selected_row = data[data['Label'] == label_value].sample(n=1) if label_value else data.sample(n=1)
    if selected_row.empty:
        raise ValueError(f"No data found for label value {label_value}")
    
    selected_coords = selected_row[['Latitude', 'Longitude']].values
    nbrs = NearestNeighbors(n_neighbors=k*k, algorithm='ball_tree').fit(data[['Latitude', 'Longitude']].values)
    subgraph_indices = nbrs.kneighbors(selected_coords)[1][0]
    subgraph_data = data.iloc[subgraph_indices]   # Get the indices of the k*k nearest neighbours

    node_features = torch.tensor(subgraph_data[['Latitude', 'Longitude', 'Diviner', 'LOLA', 'M3', 'MiniRF', 'Elevation']].values, dtype=torch.float32)
    node_labels = torch.tensor(subgraph_data['Label'].values, dtype=torch.float32)

    target_idx = subgraph_indices.tolist().index(selected_row.index[0])
    edge_index_list = []

    for i in range(k*k):
        for j in range(k*k):
            if i != j:
                edge_index_list.append([i, j])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    self_loops = torch.arange(k*k, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    return node_features, edge_index, node_labels, target_idx


def plot_metrics(num_epochs, 
                 train_losses, val_losses, test_loss,
                 val_mses, val_r2s, test_mse, test_r2, 
                 save_path='../figs/training_metrics.png'):
    
    num_epochs = num_epochs.cpu().numpy() if isinstance(num_epochs, torch.Tensor) else num_epochs
    train_losses = train_losses.cpu().numpy() if isinstance(train_losses, torch.Tensor) else train_losses
    test_loss = test_loss.cpu().numpy() if isinstance(test_loss, torch.Tensor) else test_loss
    val_losses = val_losses.cpu().numpy() if isinstance(val_losses, torch.Tensor) else val_losses
    val_mses = val_mses.cpu().numpy() if isinstance(val_mses, torch.Tensor) else val_mses
    val_r2s = val_r2s.cpu().numpy() if isinstance(val_r2s, torch.Tensor) else val_r2s
    test_mse = test_mse.cpu().numpy() if isinstance(test_mse, torch.Tensor) else test_mse
    test_r2 = test_r2.cpu().numpy() if isinstance(test_r2, torch.Tensor) else test_r2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    ax1.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    ax1.scatter(num_epochs, test_loss, color='blue', label='Test Loss', zorder=5)
    ax1.annotate(f'{test_loss:.4f}', (num_epochs, test_loss), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(range(1, num_epochs + 1), val_mses, label='Validation MSE')
    ax2.plot(range(1, num_epochs + 1), val_r2s, label='Validation R²')
    ax2.scatter(num_epochs, test_mse, color='blue', label='Test MSE', zorder=5)
    ax2.scatter(num_epochs, test_r2, color='blue', label='Test R²', zorder=5)
    ax2.annotate(f'{test_mse:.4f}', (num_epochs, test_mse), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    ax2.annotate(f'{test_r2:.4f}', (num_epochs, test_r2), textcoords="offset points", xytext=(0,-10), ha='center', color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric')
    ax2.set_title('Validation MSE and R²')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def change_grid_resolution(df, factor):
    """
    Change the resolution of the coordinate mesh grid.

    Parameters:
    df (pd.DataFrame): The input dataframe with latitude and longitude columns.
    factor (int): The factor by which to decrease the resolution.
                  (e.g., 2 to double the resolution, 3 to triple the resolution)

    Returns:
    pd.DataFrame: The dataframe with decreased resolution.
    """

    original_resolution = 240
    new_resolution = original_resolution * factor
    print(f"Grid resolution changed from {original_resolution} to {new_resolution} meters.")
    
    # Sorting by latitude and then by longitude
    df_sorted = df.sort_values(by=['Latitude', 'Longitude']).reset_index(drop=True)
    
    # Removing rows to thin out the grid in the longitude direction
    df_thinned_long = df_sorted.groupby('Latitude').apply(lambda x: x.iloc[::factor, :]).reset_index(drop=True)
    
    # Removing groups to thin out the grid in the latitude direction
    df_thinned = df_thinned_long.iloc[::factor, :].reset_index(drop=True)
    
    return df_thinned