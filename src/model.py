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

output_dim = 1  # binary classification for thumbs up or down
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for classification as a thumbs up

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
        return torch.sigmoid(out)


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

    model = FeedforwardNeuralNetModel(input_dim, 100, output_dim)
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