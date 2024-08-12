import torch
import torch.optim as optim
import time
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from models import FCNN
from custom_loss import FocalLoss, HuberLoss
from utils import create_FCNN_loader, gen_rand_data, stratified_split_data, plot_metrics
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

def train(args):
    start_time = time.time()

    rand_state = 42
    torch.manual_seed(rand_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = 1    

    # Example data
    # npoints = 1_000
    # labeled_data = gen_rand_data(npoints, rand_state)

    # Load data
    labeled_data = load_data(args.data_path)

    # Extract features and targets and convert to pt tensor
    # feature_df = labeled_data.loc[:, labeled_data.columns != "Label"]
    feature_df = labeled_data[["LOLA", "Diviner", "M3", "MiniRF"]]
    features = feature_df.values
    print(f"Features: {feature_df.columns.tolist()}")

    targets = labeled_data["Label"].values
    input_dim = features.shape[1]
    print(f"Input dimension: {input_dim}")
    sys.stdout.flush()

    train_features, val_features, test_features, train_targets, val_targets, test_targets = stratified_split_data(features, targets)

    train_loader = create_FCNN_loader(train_features, train_targets, device, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = create_FCNN_loader(test_features, test_targets, device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    model = FCNN(input_dim, args.hidden_dim, output_dim, args.dropout_rate)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    # criterion = FocalLoss(alpha=1, gamma=2)
    criterion = HuberLoss(delta=1.0)

    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []

    print(f"Entering training loop after {time.time() - start_time :.2f} seconds")
    sys.stdout.flush()
    # Training loop
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{args.num_epochs} started after {time.time() - epoch_start :.2f} seconds")
        sys.stdout.flush()
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_mse, val_r2 = validate(device, model, criterion, test_loader)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_r2s.append(val_r2)

        print(f'Epoch [{epoch+1:02d}/{args.num_epochs}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Val mse: {val_mse:.4f}, Val R²: {val_r2:.4f}, Time: {time.time() - epoch_start:.2f}s')

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
    plot_metrics(args.num_epochs, train_losses, val_losses, val_mses, val_r2s, test_mse, test_mae, test_r2, save_path='../figs/training_metrics_FCNN.png')

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
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for data loaders.')
    return parser.parse_args()

if __name__ == '__main__':
    main()