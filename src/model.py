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
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
)

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
    # Load all data files first (without processing)
    testing_files_data = {}
    training_files_data = {}

    print("=== Loading Testing Data Files ===")
    for file in os.listdir(TESTING_DATA_PATH):
        if file.endswith(".json"):
            file_path = os.path.join(TESTING_DATA_PATH, file)
            print(f"Loading file: {file}")
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    testing_files_data[file] = data
                    print(f"  {file}: {len(data)} samples")

    print("\n=== Loading Training Data Files ===")
    for file in os.listdir(TRAINING_DATA_PATH):
        if file.endswith(".json"):
            file_path = os.path.join(TRAINING_DATA_PATH, file)
            print(f"Loading file: {file}")
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    training_files_data[file] = data
                    print(f"  {file}: {len(data)} samples")

    # Find minimum length across all files
    all_lengths = []
    for file, data in testing_files_data.items():
        all_lengths.append(len(data))
    for file, data in training_files_data.items():
        all_lengths.append(len(data))

    if not all_lengths:
        print("ERROR: No valid data files found.")
        return

    min_length = min(all_lengths)
    print(f"\n=== Data Balancing ===")
    print(f"Minimum file length: {min_length} samples")
    print(f"Will extract {min_length} samples from each file for balanced dataset")

    # Now process the data with balanced sampling
    testing_data = []
    training_data = []

    print("\n=== Processing Testing Data ===")
    for file, data in testing_files_data.items():
        print(f"Processing {file}: extracting {min_length} from {len(data)} samples")
        # Take only up to min_length samples from each file
        balanced_data = data[:min_length]

        for item in balanced_data:
            if isinstance(item, dict) and "handData" in item and "label" in item:
                # Create flat list: handData + label
                row = item["handData"] + [item["label"]]
                testing_data.append(row)

    print("\n=== Processing Training Data ===")
    for file, data in training_files_data.items():
        print(f"Processing {file}: extracting {min_length} from {len(data)} samples")
        # Take only up to min_length samples from each file
        balanced_data = data[:min_length]

        for item in balanced_data:
            if isinstance(item, dict) and "handData" in item and "label" in item:
                # Create flat list: handData + label
                row = item["handData"] + [item["label"]]
                training_data.append(row)

    if not training_data or not testing_data:
        print(
            "ERROR: No valid data loaded. Check your JSON file format and run modify_data.py first."
        )
        return

    # Convert to numpy arrays
    training_data = np.array(training_data)
    testing_data = np.array(testing_data)

    print(f"\n=== Final Dataset Statistics ===")
    print(f"Training data shape: {training_data.shape}")
    print(f"Testing data shape: {testing_data.shape}")
    print(f"Total combined data: {len(training_data) + len(testing_data)} samples")

    # Shuffle data
    np.random.shuffle(training_data)
    np.random.shuffle(testing_data)

    # output stats
    print(
        f"Training data: {len(training_data)} samples ({len(training_data)/(len(training_data) + len(testing_data))*100:.1f}%)"
    )
    print(
        f"Test data: {len(testing_data)} samples ({len(testing_data)/(len(testing_data) + len(training_data))*100:.1f}%)"
    )

    # Create directories if they don't exist
    os.makedirs("src/train_data", exist_ok=True)
    os.makedirs("src/test_data", exist_ok=True)

    # Save the split data
    torch.save(training_data, "src/train_data/train_split.pt")
    torch.save(testing_data, "src/test_data/test_split.pt")

    batch_size = 16  # Keep consistent
    num_epochs = 10  # Simplified epoch calculation

    # Prepare data loaders
    X_train = torch.tensor(training_data[:, :-1], dtype=torch.float32)
    y_train = torch.tensor(training_data[:, -1], dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=batch_size
    )

    X_test = torch.tensor(testing_data[:, :-1], dtype=torch.float32)
    y_test = torch.tensor(testing_data[:, -1], dtype=torch.long)
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=False, batch_size=batch_size
    )

    model = FeedforwardNeuralNetModel(input_dim, 100, output_dim)
    criterion = nn.BCELoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    iter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for i, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

        # Evaluation after each epoch
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for X, Y in test_loader:
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                test_total += Y.size(0)
                test_correct += (predicted == Y).sum().item()
                all_predictions.extend(predicted.numpy())
                all_labels.extend(Y.numpy())

        train_accuracy = 100 * correct / total
        test_accuracy = 100 * test_correct / test_total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Test Acc: {test_accuracy:.2f}%")

        model.train()

    # Final evaluation with classification report
    print("\n=== Final Model Performance ===")
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Closed Fist", "Finger Gun", "Peace Sign", "Thumbs Up"],
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
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy_input,
        SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx",
    )

    print(f"\nModel weights saved to {SAVE_MODEL_PATH + SAVE_MODEL_FILENAME}")


if __name__ == "__main__":
    main()
