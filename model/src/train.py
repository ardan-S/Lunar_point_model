import torch
import torch.optim as optim
import time

from model import PointRankingModel
from custom_loss import FocalLoss
from utils import create_data_loader, gen_rand_data, stratified_split_data
from evaluate import evaluate, validate


"""
Considerations for training techniques:
Early Stopping: Monitor validation performance and stop training when performance stops improving to prevent overfitting.
Cross-Validation: Use k-fold cross-validation to ensure the model's performance is consistent across different subsets of the data.
Batch Normalization and Layer Normalization: Experiment with different normalization techniques to stabilize and accelerate training.
"""

start_time = time.time()
rand_state = 42
torch.manual_seed(rand_state)

input_dim = 6
output_dim = 1

# Hyperparameters
hidden_dim = 256
num_epochs = 30
learning_rate = 0.001
dropout_rate = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointRankingModel(input_dim, hidden_dim, output_dim, dropout_rate).to(device)

optimiser = optim.Adam(model.parameters(), lr=learning_rate)

criterion = FocalLoss(alpha=1, gamma=2)

# Example data
npoints = 1_000
labeled_data = gen_rand_data(npoints, rand_state)

# Extract features and targets and convert to pt tensor
features = labeled_data[["Longitude", "Latitude", "Diviner", "LOLA", "M3", "MiniRF"]].values
targets = labeled_data["Label"].values

train_features, val_features, test_features, train_targets, val_targets, test_targets = stratified_split_data(features, targets)

train_loader = create_data_loader(train_features, train_targets, batch_size=32, shuffle=True)
test_loader = create_data_loader(test_features, test_targets, batch_size=32, shuffle=False)

print(f"Entering training loop after {time.time() - start_time :.2f} seconds")
# Training loop
for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs).squeeze()

        loss = criterion(outputs, targets)
        val_loss, val_acc = validate(device, model, criterion, test_loader)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val loss: {val_loss:.4f}, Time: {time.time() - epoch_start:.2f}s')

print(f"Training completed in {time.time() - start_time :.2f} seconds")

# # Save the model for evaluation
# torch.save(model.state_dict(), 'point_ranking_model.pth')

# Evaluate the model
mse, mae, r2 = evaluate(device, model, test_loader)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R²): {r2:.4f}')
