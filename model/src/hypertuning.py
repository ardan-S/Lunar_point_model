
"""Adjust dropout rate, schedules, consider L1 or L2 regularization,
learning rate annealing, cyclical learning rates, or adaptive learning rates (e.g., using AdamW optimizer).


Optuna, Hyperopt, or GridSearchCV for hyperparameter tuning"""


import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import time

from models import FCNN
from custom_loss import HuberLoss
from utils import create_data_loader, gen_rand_data, stratified_split_data, plot_metrics
from evaluate import validate

def load_data(data_path):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Directory {data_path} not found")
    
    all_csvs = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    filepaths = [os.path.join(data_path, f) for f in all_csvs]

    data = pd.concat([pd.read_csv(file) for file in filepaths], ignore_index=True)

    return data

def train_and_validate(args, delta):
    rand_state = 42
    torch.manual_seed(rand_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = 1    

    # Example data
    npoints = 1_000
    labeled_data = gen_rand_data(npoints, rand_state)

    # Load data
    # labeled_data = load_data(args.data_path)

    # Extract features and targets and convert to pt tensor
    feature_df = labeled_data.loc[:, labeled_data.columns != "Label"]
    features = feature_df.values

    targets = labeled_data["Label"].values
    input_dim = features.shape[1]

    train_features, val_features, test_features, train_targets, val_targets, test_targets = stratified_split_data(features, targets)

    train_loader = create_data_loader(train_features, train_targets, batch_size=args.batch_size, shuffle=True)
    val_loader = create_data_loader(val_features, val_targets, batch_size=args.batch_size, shuffle=False)

    model = FCNN(input_dim, args.hidden_dim, output_dim, args.dropout_rate).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = HuberLoss(delta=delta)

    for epoch in range(args.num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    val_loss, val_mse, val_r2 = validate(device, model, criterion, val_loader)
    return val_loss, val_mse, val_r2

def hyperparameter_search(args):
    deltas = np.linspace(0.001, 10, 100)  # Define a range of delta values to search
    best_delta = None
    best_val_loss = float('inf')

    results = []

    for delta in deltas:
        val_loss, val_mse, val_r2 = train_and_validate(args, delta)
        results.append((delta, val_loss, val_mse, val_r2))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mse = val_mse
            best_val_r2 = val_r2
            best_delta = delta

    print(f"Best Delta: {best_delta}, Best Val Loss: {best_val_loss:.4f}, Best Val MSE: {best_val_mse:.4f}, Best Val R²: {best_val_r2:.4f}")

    return best_delta, results

def main():
    args = parse_arguments()
    best_delta, results = hyperparameter_search(args)
    print("Hyperparameter search completed.")
    print(f"Best delta: {best_delta}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for PointRankingModel.')
    parser.add_argument('--data_path', type=str, default='../../data/Combined_CSVs', help='Path to the input data file.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the hidden layer.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders.')
    return parser.parse_args()

if __name__ == '__main__':
    main()
