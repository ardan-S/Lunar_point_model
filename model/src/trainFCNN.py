import torch
import torch.optim as optim
import torch.nn as nn
import time
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

from models import FCNN
from utils import load_data, create_FCNN_loader, stratified_split_data, plot_metrics, balanced_sample
from evaluate import evaluate, validate


def setup_FCNN_data(args):
    """
    Function to set up the data for the FCNN model.
    Loads the data, balances the classes, and splits the data into training, validation, and test sets.
    """
    rand_state = 42
    torch.manual_seed(rand_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_data = load_data(args.data_path)
    labeled_data = balanced_sample(labeled_data, 'Label', 0.001, random_state=rand_state)

    features = labeled_data[["LOLA", "Diviner", "M3", "MiniRF", "Elevation", "Latitude", "Longitude"]]

    targets = labeled_data["Label"]
    input_dim = features.shape[1]

    train_features, val_features, test_features, train_targets, val_targets, test_targets = stratified_split_data(features, targets)

    return device, input_dim, train_features, train_targets, val_features, val_targets, test_features, test_targets


def setup_FCNN_loader(train_features, train_targets, val_features, val_targets, test_features, test_targets, device, args):
    """
    Function to setup the data loaders for the FCNN model.
    Fits the scalers on the training data and creates the loaders for the training, validation, and test sets.
    """
    standardise_scalar = StandardScaler().fit(train_features)   # Fit the scalers on the training data only
    normalise_scalar = MinMaxScaler().fit(train_features)

    joblib.dump(standardise_scalar, '../saved_models/standardise_scalar_FCNN.joblib')
    joblib.dump(normalise_scalar, '../saved_models/normalise_scalar_FCNN.joblib')

    train_loader = create_FCNN_loader(train_features, train_targets, device, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, standardise_scalar=standardise_scalar, normalise_scalar=normalise_scalar)
    test_loader = create_FCNN_loader(test_features, test_targets, device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, standardise_scalar=standardise_scalar, normalise_scalar=normalise_scalar)
    val_loader = create_FCNN_loader(val_features, val_targets, device, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, standardise_scalar=standardise_scalar, normalise_scalar=normalise_scalar)

    # Raise if any of the loaders contain nans
    for inputs, targets in train_loader:
        if torch.isnan(inputs).any():
            raise ValueError("Found NaN values in the inputs during training.")
        if torch.isnan(targets).any():
            raise ValueError("Found NaN values in the targets during training.")

    for inputs, targets in val_loader:
        if torch.isnan(inputs).any():
            raise ValueError("Found NaN values in the inputs during validation.")
        if torch.isnan(targets).any():
            raise ValueError("Found NaN values in the targets during validation.")

    for inputs, targets in test_loader:
        if torch.isnan(inputs).any():
            raise ValueError("Found NaN values in the inputs during testing.")
        if torch.isnan(targets).any():
            raise ValueError("Found NaN values in the targets during testing.")
    return train_loader, val_loader, test_loader


def setup_FCNN_model(input_dim, args, device):
    """
    Function to set up the FCNN model, criterion, optimizer, and scaler.
    Defines the model, loss function, optimizer, and scaler for mixed precision training.
    """
    model = FCNN(input_dim, args.hidden_dim, 1, args.dropout_rate).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=args.beta)
    scaler = torch.amp.GradScaler()    # Initialise GradScaler for mixed precision training

    return model, criterion, optimiser, scaler


def train_FCNN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, model_save_path=None, img_save_path=None):
    """
    Function to train the FCNN model.
    Trains the model on the training set and validates on the validation set.
    Evaluates on the test set and saves the model and training metrics if specified.
    """
    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []

    iter = 0

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        iter += 1

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()

            # Use autocast to enable mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

            if torch.isnan(outputs).any():
                raise ValueError("Model outputs contain NaN values during validation")

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

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1:03d}/{args.num_epochs}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Val mse: {val_mse:.4f}, Val R²: {val_r2:.4f}, Time: {time.time() - epoch_start:.2f}s')

    test_loss, test_mse, test_r2 = evaluate(device, model, criterion, test_loader)

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        file_path = model_save_path.replace('.pth', '.txt')
        content = f"""
        Parameters of saved FCNN:
        Hidden dimension: {args.hidden_dim}
        Epochs: {args.num_epochs}
        Learning rate: {args.learning_rate}
        Dropout rate: {args.dropout_rate}
        Batch size: {args.batch_size}
        Beta: {args.beta}
        Weight decay: {args.weight_decay}

        Test set:
        Mean Squared Error (MSE): {test_mse:.4f}
        Loss: {test_loss:.4f}
        R-squared (R²): {test_r2:.4f}

        Saved on {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Model saved at {model_save_path}")

    if img_save_path:
        plot_metrics(iter, train_losses, val_losses, test_loss, val_mses, val_r2s, test_mse, test_r2, save_path=img_save_path)

    return model, test_loss, test_mse, test_r2


def main():
    """
    Main function to train the FCNN model.
    Calls the setup functions, trains the model, and prints the test metrics.
    Takes a random line from the test loader and prints the true labels and predictions.
    """
    args = parse_arguments()
    start_time = time.time()
    device, input_dim, train_features, train_targets, val_features, val_targets, test_features, test_targets = setup_FCNN_data(args)
    print(f"Data setup completed after {(time.time() - start_time) / 60 :.2f} mins")
    train_loader, val_loader, test_loader = setup_FCNN_loader(train_features, train_targets, val_features, val_targets, test_features, test_targets, device, args)
    print(f"Loader setup completed after {(time.time() - start_time) / 60 :.2f} mins")
    model, criterion, optimiser, scaler = setup_FCNN_model(input_dim, args, device)
    print(f"Model setup completed after {(time.time() - start_time) / 60 :.2f} mins")
    model, test_loss, test_mse, test_r2 = train_FCNN(device, model, criterion, optimiser, scaler, train_loader, val_loader, test_loader, args, img_save_path='../figs/training_metrics_FCNN.png', model_save_path='../saved_models/FCNN.pth')
    print(f"Training completed after {(time.time() - start_time) / 60 :.2f} mins")
    print("\nTest set:")
    print(f'Mean Squared Error (MSE): {test_mse:.4f}')
    print(f'Loss: {test_loss:.4f}')
    print(f'R-squared (R²): {test_r2:.4f}\n')

    # Get random line from test loader
    random_line = next(iter(test_loader))
    inputs, targets = random_line
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs).squeeze()

    targets_np = targets.cpu().detach().numpy()[:5]
    outputs_np = outputs.cpu().detach().numpy()[:5]
    print(f"True labels: {targets_np}")
    print(f"Predictions: {outputs_np}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a PointRankingModel.')
    parser.add_argument('--data_path', type=str, default='../../data/Combined_CSVs', help='Path to the input data file.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the hidden layer.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders.')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta for the smooth L1 loss.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for the optimizer.')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for data loaders.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
