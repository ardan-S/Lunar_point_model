import torch
import torch.optim as optim
import time
import argparse
# import os
import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from models import FCNN
from custom_loss import FocalLoss, HuberLoss
from utils import load_data, create_FCNN_loader, gen_rand_data, stratified_split_data, plot_metrics, change_grid_resolution
from evaluate import evaluate, validate


"""
Considerations for training techniques:
Early Stopping: Monitor validation performance and stop training when performance stops improving to prevent overfitting.
Cross-Validation: Use k-fold cross-validation to ensure the model's performance is consistent across different subsets of the data.
Batch Normalization and Layer Normalization: Experiment with different normalization techniques to stabilize and accelerate training.
"""

def setup_FCNN_data(args):
    rand_state = 42
    torch.manual_seed(rand_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = 1    

    # Example data
    # npoints = 1_000
    # labeled_data = gen_rand_data(npoints, rand_state)

    # Load data
    labeled_data = load_data(args.data_path)
    original_size = labeled_data.shape[0]
    print(f"Size of original dataset: {original_size}")

    labeled_data = labeled_data[labeled_data["Latitude"] < 0]   # Only train on southern hemisphere and really close to the pole
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.0001, random_state=rand_state)
    _, test_index = next(sss.split(labeled_data, labeled_data["Label"]))
    labeled_data = labeled_data.iloc[test_index]
    print(f"Size of selected dataset: {labeled_data.shape[0]}")
    
    """
    Changing the grid resolution by a factor of 4 to reduce the load on the model and speed up training.
    OR drop a pole.
    """

    # Extract features and targets and convert to pt tensor
    features = labeled_data[["LOLA", "Diviner", "M3", "MiniRF"]]
    print(f"Features: {features.columns.tolist()}")

    targets = labeled_data["Label"]
    input_dim = features.shape[1]

    train_features, val_features, test_features, train_targets, val_targets, test_targets = stratified_split_data(features, targets)

    return device, input_dim, train_features, train_targets, val_features, val_targets, test_features, test_targets


def setup_FCNN_loader(train_features, train_targets, val_features, val_targets, test_features, test_targets, device, args):
    train_loader = create_FCNN_loader(train_features, train_targets, device, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = create_FCNN_loader(test_features, test_targets, device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    val_loader = create_FCNN_loader(val_features, val_targets, device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    
    return train_loader, val_loader, test_loader


def setup_FCNN_model(input_dim, args, device):
    output_dim = 1
    model = FCNN(input_dim, args.hidden_dim, output_dim, args.dropout_rate).to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)        # Could consider using DistributedDataParallel but requires more setup
    
    print(f"Training on {device} with {torch.cuda.device_count()} GPUs")

    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler()    # Initialise GradScaler for mixed precision training 

    return model, criterion, optimiser, scaler


def train_FCNN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, model_save_path=None, img_save_path=None):
    start_time = time.time()
    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []

    print(f"\nEntering training loop")
    sys.stdout.flush()
    # Training loop
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()

            # Use autocast to enable mixed precision training
            with torch.amp.autocast('cuda'):
            # with torch.no_grad():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

            # loss.backward()
            # optimiser.step()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_mse, val_r2 = validate(device, model, criterion, val_loader)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_r2s.append(val_r2)

        print(f'Epoch [{epoch+1:02d}/{args.num_epochs}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Val mse: {val_mse:.4f}, Val R²: {val_r2:.4f}, Time: {time.time() - epoch_start:.2f}s')
        sys.stdout.flush()

    print(f"Training completed in {(time.time() - start_time) / 60 :.2f} mins")
    test_mse, test_mae, test_r2 = evaluate(device, model, test_loader)

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    if img_save_path:
        plot_metrics(args.num_epochs, train_losses, val_losses, val_mses, val_r2s, test_mse, test_mae, test_r2, save_path=img_save_path)

    return model, test_mse, test_mae, test_r2

def main():
    args = parse_arguments()
    start_time = time.time()
    device, input_dim, train_features, train_targets, val_features, val_targets, test_features, test_targets = setup_FCNN_data(args)
    print(f"Data setup completed in {(time.time() - start_time) / 60 :.2f} mins")
    train_loader, val_loader, test_loader = setup_FCNN_loader(train_features, train_targets, val_features, val_targets, test_features, test_targets, device, args)
    print(f"Loader setup completed in {(time.time() - start_time) / 60 :.2f} mins")
    model, criterion, optimiser, scaler = setup_FCNN_model(input_dim, args, device)
    print(f"Model setup completed in {(time.time() - start_time) / 60 :.2f} mins")
    model, test_mse, test_mae, test_r2 = train_FCNN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args)
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
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for data loaders.')
    return parser.parse_args()

if __name__ == '__main__':
    main()