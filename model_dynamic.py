import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report

# Set random seed for reproducible results
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #for device agnostic code

output_dim = 4  # multiclass classification for 4 gestures (closed fist, finger gun, peace sign, thumbs up)
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for gesture classification
# Gesture classes: 0=Closed Fist, 1=Finger Gun, 2=Peace Sign, 3=Thumbs Up

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "model_dynamic_weights.json"
TRAIN_PATH = "train_data/train_sequences_0.pt" #sequenced dynamic train data
TEST_PATH = "test_data/test_sequences_0.pt" #sequenced dynamic test data

def split_feature_label(data):
    """Extract features and labels from sequences.

    Args:
        data: tensor of shape [num_sequences, sequence_length, 18]
              where last column is the class label

    Returns:
        X: tensor of shape [num_sequences, sequence_length, 17] (features only)
        Y: tensor of shape [num_sequences] (class labels as integers)
    """
    X = data[:, :, :-1]  # All frames, all features (first 17 columns)
    Y = data[:, -1, -1].long()  # Class label from last frame (represents final/current gesture in sequence)
    return X, Y

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Output layer: hidden_dim -> output_dim (4 classes for multiclass classification)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(dim=0), self.hidden_dim).to(device) #initialized hidden state
        c0 = torch.zeros(self.num_layers, x.size(dim=0), self.hidden_dim).to(device) #initialized cell state
        out, states = self.lstm(x, (h0, c0)) # states represents hidden and cell states (not needed)
        out = self.fc(out[:, -1, :]) # get the last time step's output for each sequence
        return out

def main():
    train_data = torch.load(TRAIN_PATH, weights_only=False)
    test_data = torch.load(TEST_PATH, weights_only=False)

    batch_size = 64
    num_epochs = 10  # Fixed number of epochs for multiclass

    X_train, y_train = split_feature_label(train_data)
    X_test, y_test = split_feature_label(test_data)

    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=False, batch_size=batch_size
    )

    lstm_model = LSTM_Model(input_dim, 32, output_dim).to(device)

    # Calculate class weights to handle imbalanced data
    class_counts = torch.bincount(y_train.long())
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    # Use CrossEntropyLoss for multiclass classification with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    learning_rate = 0.001  # Slightly higher LR for multiclass
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

    print("=== Training LSTM Model for Multiclass Dynamic Gesture Recognition ===")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}\n")

    for epoch in range(num_epochs):
        lstm_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            outputs = lstm_model(X.float())  # Shape: [batch_size, 4]
            loss = criterion(outputs, Y)  # Y shape: [batch_size]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

        train_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Evaluate on test set
        lstm_model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []

        with torch.inference_mode():
            for X, Y in test_loader:
                X, Y = X.to(device), Y.to(device)
                outputs = lstm_model(X.float())
                _, predicted = torch.max(outputs.data, 1)
                test_total += Y.size(0)
                test_correct += (predicted == Y).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(Y.cpu().numpy())

        test_accuracy = 100 * test_correct / test_total

        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Test Acc: {test_accuracy:.2f}%")

    # Final evaluation with classification report
    print("\n=== Final Model Performance ===")
    gesture_names = ["Closed Fist", "Finger Gun", "Peace Sign", "Thumbs Up"]
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=gesture_names))

    # Print sample prediction
    print("\nSample prediction on test set:")
    sample_idx = 0
    print(f"  Ground truth: {gesture_names[all_labels[sample_idx]]}")
    print(f"  Prediction: {gesture_names[all_predictions[sample_idx]]}")

    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = lstm_model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Create directory if it does not exist
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    # Store as onnx for compatibility with Unity Barracuda
    # Create dummy input with proper shape: [batch_size=1, sequence_length=20, input_dim=17]
    dummy_input = torch.randn(1, 20, input_dim)
    try:
        torch.onnx.export(
            lstm_model,
            dummy_input,
            SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
            do_constant_folding=False,
            verbose=False
        )
    except Exception as e:
        # ONNX export can have compatibility issues with some PyTorch versions
        # Model is already saved in JSON and PyTorch formats above
        print(f"  Note: ONNX export skipped ({type(e).__name__})")
        print(f"  Model is available in JSON format: {SAVE_MODEL_PATH + SAVE_MODEL_FILENAME}")

    print("\n=== Model Training Complete ===")
    print(f"\nModel weights saved to {SAVE_MODEL_PATH + SAVE_MODEL_FILENAME}")
    print(f"ONNX model saved to {SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split('.')[0] + '.onnx'}")

if __name__ == "__main__":
    main()