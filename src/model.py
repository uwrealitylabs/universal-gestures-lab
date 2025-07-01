import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import os
from . import utils
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

output_dim = 5  # multi-class classification for 5 classes
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for classification as a specific class

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "model_multi_class_weights.json"


# Model
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Define the feedforward neural network architecture
        #       with 4 hidden layers.

        # input -> layer 1
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()

        # layer 1 -> 2
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()

        # layer 2 -> 3
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()

        # layer 3 -> 4
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()

        # layer 4 -> output
        self.fc5 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Assuming x is of shape (batch_size, input_dim)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x  # raw logits for CrossEntropyLoss


# Data set
def split_feature_label(data):
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X, self.Y = split_feature_label(data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Loader fn
def load_data(dataset, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main():
    train_path = "src/train_data/train_0.pt"
    test_path = "src/test_data/test_0.pt"
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    batch_size = 64
    n_iters = len(train_data) * 5  # 5 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))

    X_train = torch.tensor(train_data[:, :-1])
    y_train = torch.tensor(train_data[:, -1])
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=16
    )

    X_test = torch.tensor(test_data[:, :-1])
    y_test = torch.tensor(test_data[:, -1])
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=True, batch_size=16
    )

    model = FeedforwardNeuralNetModel(input_dim, None, 5)
    criterion = nn.BCELoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    iter = 0

    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            Y = Y.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, Y.float())
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                correct = 0
                total = 0
                all_labels = []
                all_probs = []
                for X, Y in test_loader:
                    outputs = model(X.float())
                    probs = outputs.detach().numpy().flatten()
                    predicted = (outputs > detect_threshold).float()
                    total += Y.size(0)
                    correct += (predicted == Y.view(-1, 1)).sum().item()
                    all_labels.extend(Y.numpy())
                    all_probs.extend(probs)

                accuracy = 100 * correct / total
                auc_roc = roc_auc_score(all_labels, all_probs)
                precision, recall, _ = precision_recall_curve(all_labels, all_probs)
                auc_pr = auc(recall, precision)

                # Example: Log metrics to the database
                model_type = (
                    "binary classification"  # Adjust based on your specific model type
                )

                # Logging disabled for package release
                # utils.log_training_metrics(auc_pr, auc_roc, loss.item(), model_type)

                print(
                    "Iteration: {}. Loss: {}. Accuracy: {}. AUC-ROC: {:.4f}. AUC-PR: {:.4f}".format(
                        iter, loss.item(), accuracy, auc_roc, auc_pr
                    )
                )

    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Create directory if it does not exist
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    # Store as onnx for compatibility with Unity Barracuda
    onnx_program = torch.onnx.dynamo_export(model, torch.randn(1, input_dim))
    onnx_program.save(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx")

    print("\n--- Model Training Complete ---")
    print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)


if __name__ == "__main__":
    main()
