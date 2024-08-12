import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader  # Avoid name conflict with PyTorch DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_processing.label import apply_labels


def create_FCNN_loader(inputs, targets, device, batch_size=32, shuffle=True, num_workers=1):
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

    # Normalize the input features (mean=0, std=1)
    standardise_scalar = StandardScaler()
    normalise_scalar = MinMaxScaler()
    inputs = normalise_scalar.fit_transform(standardise_scalar.fit_transform(inputs))
    targets = normalise_scalar.fit_transform(standardise_scalar.fit_transform(targets.reshape(-1, 1))).reshape(-1)

    # Convert inputs and targets to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32) if not torch.is_tensor(inputs) else inputs
    targets = torch.tensor(targets, dtype=torch.float32) if not torch.is_tensor(targets) else targets

    # Create dataset and data loader
    dataset = TensorDataset(inputs, targets)
    if device == 'cuda':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

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
        data_loader = GeoDataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
        print("End of data loader")
    else:
        data_loader = GeoDataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    print("Data loader created")
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

    # Split data into training and validation sets using regular splits
    train_features, test_val_features, train_targets, test_val_targets = train_test_split(
        features, targets, test_size=0.2, random_state=rand_state)
    test_features, val_features, test_targets, val_targets = train_test_split(
        test_val_features, test_val_targets, test_size=0.5, random_state=rand_state)

    return train_features, val_features, test_features, train_targets, val_targets, test_targets


def plot_metrics(num_epochs, train_losses, val_losses, val_mses, val_r2s, test_mse, test_mae, test_r2, save_path='../figs/training_metrics.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    ax1.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    ax1.scatter(num_epochs, test_mae, color='blue', label='Test Loss', zorder=5)
    ax1.annotate(f'{test_mae:.4f}', (num_epochs, test_mae), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(range(1, num_epochs + 1), val_mses, label='Validation MSE')
    ax2.plot(range(1, num_epochs + 1), val_r2s, label='Validation R²')
    ax2.scatter(num_epochs, test_mse, color='blue', label='Test MSE', zorder=5)
    ax2.scatter(num_epochs, test_r2, color='blue', label='Test R²', zorder=5)
    ax2.annotate(f'{test_mse:.4f}', (num_epochs, test_mse), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    ax2.annotate(f'{test_r2:.4f}', (num_epochs, test_r2), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric')
    ax2.set_title('Validation MSE and R²')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()