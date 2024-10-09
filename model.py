import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# changed to multi-class classification
output_dim = 5  # 5 classes
input_dim = 17  # 17 features
detect_threshold = 0.7  # dont need anymore

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "model_weights.json"


# Model
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.sigmoid(out)
        # Linear function (readout)
        out = self.fc2(out)
        # Returning raw logits without softmax (CrossEntropyLoss can deal w it)
        return out  


# Data set
def split_feature_label(data):
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X, self.Y = split_feature_label(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Loader fn
def load_data(dataset, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main():
    train_path = "train_data/train_0.pt"
    test_path = "test_data/test_0.pt"
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    batch_size = 64
    n_iters = len(train_data) * 5  # 5 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))

    X_train = torch.tensor(train_data[:, :-1])
    y_train = torch.tensor(train_data[:, -1], dtype=torch.long)  
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=16
    )

    X_test = torch.tensor(test_data[:, :-1])
    y_test = torch.tensor(test_data[:, -1], dtype=torch.long)  
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=True, batch_size=16
    )

    model = FeedforwardNeuralNetModel(input_dim, 100, output_dim)
    
    # Using CrossEntropyLoss 
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    iter = 0

    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients before the backward pass
            outputs = model(X.float())  # Forward pass
            loss = criterion(outputs, Y)  # Loss calculation for multi-class stuff
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimizer step
            iter += 1

            if iter % 500 == 0:
                correct = 0
                total = 0
                all_labels = []
                all_predictions = []
                for X, Y in test_loader:
                    outputs = model(X.float())  # Forward pass for test set
                    _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest probability
                    total += Y.size(0)
                    correct += (predicted == Y).sum().item()  # Compare predictions with actual labels
                    all_labels.extend(Y.numpy())
                    all_predictions.extend(predicted.numpy())

                accuracy = 100 * correct / total
                print(f"Iteration: {iter}. Loss: {loss.item():.4f}. Accuracy: {accuracy:.2f}%")

    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    print("\n--- Model Training Complete ---")
    print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)


if __name__ == "__main__":
    main()