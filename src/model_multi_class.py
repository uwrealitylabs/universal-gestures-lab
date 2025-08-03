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

output_dim = 4  # Changed to 4 classes (0,1,2,3)
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for classification as a specific class

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "model_multi_class_weights.json"

TESTING_DATA_PATH = "src/test_data/"
TRAINING_DATA_PATH = "src/train_data/"


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

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Unique labels in training: {torch.unique(y_train)}")

    # For multi-class classification:
    model = FeedforwardNeuralNetModel(input_dim, None, output_dim)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss instead of BCELoss
    learning_rate = 0.001  # Slightly higher learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Use Adam instead of SGD

    # Training loop
    model.train()
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

    # Print out a sample vector for a randome test
    sample_index = np.random.randint(0, len(testing_data))
    sample_vector = testing_data[sample_index, :-1]
    sample_label = testing_data[sample_index, -1]
    print(f"\nSample vector (index {sample_index}): {sample_vector}")
    print(f"Sample label: {sample_label}")
    print()
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(sample_vector, dtype=torch.float32).unsqueeze(0)
        logits = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        print(f"Model prediction for sample: {predicted_class}")


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
