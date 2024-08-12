import torch
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def validate(device, model, criterion, data_loader):
    model.eval()

    val_loss = 0.0
    total_samples = 0

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, Data):
                inputs, targets = batch.x.to(device), batch.y.to(device)
                edge_index = batch.edge_index.to(device)
                outputs = model(inputs, edge_index).squeeze()
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    val_loss /= total_samples
    val_mse = mean_squared_error(all_targets, all_outputs)
    val_r2 = r2_score(all_targets, all_outputs)

    return val_loss, val_mse, val_r2


def evaluate(device, model, test_loader):
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, Data):
                inputs, targets = batch.x.to(device), batch.y.to(device)
                edge_index = batch.edge_index.to(device)
                outputs = model(inputs, edge_index).squeeze()
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs).squeeze()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    return mse, mae, r2
