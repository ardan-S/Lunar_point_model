import itertools
import argparse
import contextlib
import sys
import os

from trainFCNN import setup_FCNN_data, setup_FCNN_loader, setup_FCNN_model, train_FCNN


@contextlib.contextmanager
def suppress_stdout():
    """
    Function to suppress stdout during hyperparameter tuning.
    """
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def hyperparameter_tuning(args):
    """
    Function to perform hyperparameter tuning for the FCNN model.
    """
    # Define the hyperparameter grid
    batch_sizes = [131072]
    learning_rates = [1e-4, 5e-5, 1e-5]
    dropout_rates = [0.1, 0.15, 0.2]
    betas = [0.05, 0.1, 0.15]
    weight_decays = [5e-4, 1e-4, 5e-5]

    best_loss = float('inf')
    best_hyperparams = None

    device, input_dim, train_features, train_targets, val_features, val_targets, test_features, test_targets = setup_FCNN_data(args)

    # Iterate over all combinations of hyperparameters
    for batch_size, lr, dropout, beta, decay in itertools.product(batch_sizes, learning_rates, dropout_rates, betas, weight_decays):
        print(f"Training with batch size: {batch_size}, learning rate: {lr}, dropout rate: {dropout}, beta: {beta}, weight decay: {decay}")
        sys.stdout.flush()

        args.batch_size = batch_size
        args.learning_rate = lr
        args.dropout_rate = dropout
        args.beta = beta
        args.weight_decay = decay

        train_loader, val_loader, test_loader = setup_FCNN_loader(train_features, train_targets, val_features, val_targets, test_features, test_targets, device, args)
        model, criterion, optimiser, scaler = setup_FCNN_model(input_dim, args, device)

        with suppress_stdout():
            _, curr_loss, curr_mse, curr_r2 = train_FCNN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args)

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_mse = curr_mse
            best_hyperparams = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'dropout_rate': dropout,
                'beta': beta,
                'weight_decay': decay
            }
            best_r2 = curr_r2

            print(f"New best hyperparameters: {best_hyperparams}")
            print(f"New best loss: {best_loss:.4f}")

    print("\nBest hyperparameters:")
    print(f"Batch size: {best_hyperparams['batch_size']} out of {batch_sizes}")
    print(f"Learning rate: {best_hyperparams['learning_rate']} out of {learning_rates}")
    print(f"Dropout rate: {best_hyperparams['dropout_rate']} out of {dropout_rates}")
    print(f"Beta: {best_hyperparams['beta']} out of {betas}")
    print(f"Weight decay: {best_hyperparams['weight_decay']} out of {weight_decays}")

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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    best_hyperparams, best_loss, best_mse, best_r2 = hyperparameter_tuning(args)
