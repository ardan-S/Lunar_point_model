import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def validate(device, model, criterion, data_loader):
    model.eval()

    val_loss, val_acc = 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            val_acc += mean_squared_error(targets.cpu().numpy(), outputs.cpu().numpy()) * inputs.size(0)
            total_samples += inputs.size(0)

    return val_loss / total_samples, val_acc / total_samples


def evaluate(device, model, test_loader):
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).squeeze()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    return mse, mae, r2
