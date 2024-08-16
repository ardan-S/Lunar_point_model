import itertools
import argparse
import torch
import contextlib
import sys
import os

from trainFCNN import setup_FCNN_data, setup_FCNN_loader, setup_FCNN_model, train_FCNN

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def hyperparameter_tuning(args):
    # Define the hyperparameter grid
    batch_sizes = [64, 128, 256]
    learning_rates = [0.001, 0.0001, 0.00001]
    dropout_rates = [0.2, 0.3, 0.4]

    best_mse = float('inf')
    best_hyperparams = None

    device, input_dim, train_features, train_targets, val_features, val_targets, test_features, test_targets = setup_FCNN_data(args)
    
    # Iterate over all combinations of hyperparameters
    for batch_size, lr, dropout in itertools.product(batch_sizes, learning_rates, dropout_rates):
        print(f"Training with batch size: {batch_size}, learning rate: {lr}, dropout rate: {dropout}")

        args.batch_size = batch_size
        args.learning_rate = lr
        args.dropout_rate = dropout

        train_loader, val_loader, test_loader = setup_FCNN_loader(train_features, train_targets, val_features, val_targets, test_features, test_targets, device, args)
        model, criterion, optimiser, scaler = setup_FCNN_model(input_dim, args, device)

        with suppress_stdout():
            _, curr_mse, _, curr_r2 = train_FCNN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args)

        if curr_mse < best_mse:
            best_mse = curr_mse
            best_hyperparams = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'dropout_rate': dropout,
            }
            best_r2 = curr_r2

    print(f"\nBest hyperparameters:")
    print(f"Batch size: {best_hyperparams['batch_size']} out of {batch_sizes}")
    print(f"Learning rate: {best_hyperparams['learning_rate']} out of {learning_rates}")
    print(f"Dropout rate: {best_hyperparams['dropout_rate']} out of {dropout_rates}")
    print(f"Best MSE: {best_mse:.4f}")
    print(f"R² for best hyperparams: {best_r2:.4f}")

    return best_hyperparams, best_mse, best_r2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for a PointRankingModel.')
    parser.add_argument('--data_path', type=str, default='../../data/Combined_CSVs', help='Path to the input data file.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the hidden layer.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for data loaders.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    best_hyperparams, best_mse, best_r2 = hyperparameter_tuning(args)
