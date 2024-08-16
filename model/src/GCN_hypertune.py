import itertools
import argparse
import torch
import time

from trainGNN import setup_GCN_data, setup_GCN_loader, setup_GCN_model, train_GCN

def hyperparameter_tuning(args):
    # Define the hyperparameter grid
    batch_sizes = [64, 128, 256]
    learning_rates = [0.001, 0.0001, 0.00001]
    dropout_rates = [0.2, 0.3, 0.4]
    k_vals = [10, 20, 30]

    best_mse = float('inf')
    best_hyperparams = None

    device, input_dim, train_graph_data, val_graph_data, test_graph_data = setup_GCN_data(args)
    
    # Iterate over all combinations of hyperparameters
    for batch_size, lr, dropout, k in itertools.product(batch_sizes, learning_rates, dropout_rates, k_vals):
        start_time = time.time()
        print(f"Training with batch size: {batch_size}, learning rate: {lr}, dropout rate: {dropout}, k: {k}")

        args.batch_size = batch_size
        args.learning_rate = lr
        args.dropout_rate = dropout
        args.k = k

        train_loader, val_loader, test_loader = setup_GCN_loader(train_graph_data, val_graph_data, test_graph_data, device, args)
        model, criterion, optimiser, scaler = setup_GCN_model(input_dim, args, device)
        model, curr_mse, curr_mae, curr_r2 = train_GCN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, img_save_path='../figs/training_metrics_GCN.png')

        if curr_mse < best_mse:
            best_mse = curr_mse
            best_hyperparams = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'dropout_rate': dropout,
            }
            best_r2 = curr_r2
        print(f"Completed in {(time.time() - start_time) / 60 :.2f} mins\n")

    print(f"\nBest hyperparameters:")
    print(f"Batch size: {best_hyperparams['batch_size']} out of {batch_sizes}")
    print(f"Learning rate: {best_hyperparams['learning_rate']} out of {learning_rates}")
    print(f"Dropout rate: {best_hyperparams['dropout_rate']} out of {dropout_rates}")
    print(f"Best k: {best_hyperparams['k']} out of {k_vals}")
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
    best_hyperparams, best_mse = hyperparameter_tuning(args)
