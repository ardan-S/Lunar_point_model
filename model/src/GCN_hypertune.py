import itertools
import argparse
import time
import sys
import os
import contextlib

from trainGNN import setup_GCN_data, setup_GCN_loader, setup_GCN_model, train_GCN


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
    batch_sizes = [2**16]
    learning_rates = [4e-4, 3e-4, 2e-4]
    dropout_rates = [0.1, 0.15, 0.2, 0.25]
    k_vals = [15]
    betas = [0.1, 0.15, 0.2]
    weight_decays = [1e-3, 5e-4, 1e-4, 5e-5]

    best_loss = float('inf')
    best_hyperparams = None

    device, input_dim, train_graph_data, val_graph_data, test_graph_data = setup_GCN_data(args)
    
    # Iterate over all combinations of hyperparameters
    for batch_size, lr, dropout, k, beta, decay in itertools.product(batch_sizes, learning_rates, dropout_rates, k_vals, betas, weight_decays):
        start_time = time.time()
        print(f"Training with learning rate: {lr}, beta: {beta}, weight decay: {decay}")
        args.batch_size = batch_size
        args.learning_rate = lr
        args.dropout_rate = dropout
        args.k = k
        args.beta = beta
        args.weight_decay = decay

        train_loader, val_loader, test_loader = setup_GCN_loader(train_graph_data, val_graph_data, test_graph_data, device, args)
        model, criterion, optimiser, scaler = setup_GCN_model(input_dim, args, device)
        with suppress_stdout():
            model, curr_loss, curr_mse, curr_r2 = train_GCN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args)

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_mse = curr_mse
            best_hyperparams = {
                'batch_size': args.batch_size,
                'learning_rate': lr,
                'dropout_rate': dropout,
                'k': k,
                'beta': beta,
                'weight_decay': decay
            }
            best_r2 = curr_r2

            print(f"New best hyperparameters: {best_hyperparams}")
            print(f"New best loss: {best_loss:.4f}")

        print(f"Completed in {(time.time() - start_time) / 60 :.2f} mins\n")

    print(f"\nBest hyperparameters:")
    print(f"Batch size: {best_hyperparams['batch_size']} out of {args.batch_size}")
    print(f"Learning rate: {best_hyperparams['learning_rate']} out of {learning_rates}")
    print(f"Dropout rate: {best_hyperparams['dropout_rate']} out of {dropout_rates}")
    print(f"Best k: {best_hyperparams['k']} out of {k_vals}")
    print(f"Best beta: {best_hyperparams['beta']} out of {betas}")
    print(f"Best weight decay: {best_hyperparams['weight_decay']} out of {weight_decays}")

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Best MSE: {best_mse:.4f}")
    print(f"R² for best hyperparams: {best_r2:.4f}")

    return best_hyperparams, best_loss, best_mse, best_r2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for a PointRankingModel.')
    parser.add_argument('--data_path', type=str, default='../../data/Combined_CSVs', help='Path to the input data file.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the hidden layer.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for data loaders.')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbours to consider.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    best_hyperparams, best_loss, best_mse, best_r2  = hyperparameter_tuning(args)
