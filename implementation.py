import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Custom Dataset for manufacturing data
class ManufacturingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Simple Neural Network for predictive maintenance
class PredictiveMaintenanceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictiveMaintenanceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
    return np.array(predictions), np.array(actuals)

if __name__ == '__main__':
    # Generate dummy data for demonstration
    np.random.seed(42)
    torch.manual_seed(42)
    num_samples = 1000
    num_features = 10

    # Simulate manufacturing data
    X = np.random.rand(num_samples, num_features)
    y = (np.sum(X, axis=1) > 5).astype(float)  # Binary classification based on sum of features

    # Split into train and test sets
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create datasets and dataloaders
    train_dataset = ManufacturingDataset(X_train, y_train)
    test_dataset = ManufacturingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = num_features
    hidden_size = 16
    output_size = 1
    model = PredictiveMaintenanceModel(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=10)

    # Evaluate the model
    predictions, actuals = evaluate_model(model, test_loader)
    print("Predictions:", predictions[:10])
    print("Actuals:", actuals[:10])