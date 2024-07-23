import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter

import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_processing.label import apply_labels


def create_data_loader(inputs, targets, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the given dataset.

    Args:
    - inputs: Tensor of input features.
    - targets: Tensor of target labels.
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
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Create dataset and data loader
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def gen_rand_data(npoints=100, rand_state=42):
    torch.manual_seed(rand_state)
    print(f"Generating {npoints} random examples...")
    sys.stdout.flush()
    inputs = torch.randn(npoints, 6)  # 6 features per example
    print("Random examples generated")
    sys.stdout.flush()
    
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

    # # First split: train and (test + validation)
    # stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rand_state)

    # for train_index, test_val_index in stratified_split.split(features, targets):
    #     train_features, test_val_features = features[train_index], features[test_val_index]
    #     train_targets, test_val_targets = targets[train_index], targets[test_val_index]

    # # Second split: test and validation
    # stratified_split_test_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=rand_state)

    # for test_index, val_index in stratified_split_test_val.split(test_val_features, test_val_targets):
    #     test_features, val_features = test_val_features[test_index], test_val_features[val_index]
    #     test_targets, val_targets = test_val_targets[test_index], test_val_targets[val_index]

    # # Print sizes of the sets to verify
    # print(f'Train set size: {len(train_features)}')
    # print(f'Validation set size: {len(val_features)}')
    # print(f'Test set size: {len(test_features)}')

    # Split data into training and validation sets using regular splits
    train_features, test_val_features, train_targets, test_val_targets = train_test_split(
        features, targets, test_size=0.2, random_state=rand_state)
    test_features, val_features, test_targets, val_targets = train_test_split(
        test_val_features, test_val_targets, test_size=0.5, random_state=rand_state)

    return train_features, val_features, test_features, train_targets, val_targets, test_targets
