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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #for device agnostic code

output_dim = 1  # binary classification for gesture detected or not
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for gesture classification

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "model_dynamic_weights.json"
TRAIN_PATH = "train_data/train_sequences_0.pt" #sequenced dynamic train data
TEST_PATH = "test_data/test_sequences_0.pt" #sequenced dynamic test data

def split_feature_label(data):
    X = data[:, :, :-1]
    Y = data[:, -1, -1]
    return X, Y

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
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
    n_iters = len(train_data) * 5  # 5 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))

    X_train, y_train = split_feature_label(train_data)
    X_test, y_test = split_feature_label(test_data)
    
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=64
    )
    
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=True, batch_size=64
    )
    
    lstm_model = LSTM_Model(input_dim, 32, output_dim).to(device) #32 hidden layers
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate)
    iter = 0

    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            lstm_model.train()
            X, Y = X.to(device), Y.to(device)
            Y = Y.view(-1, 1)
            optimizer.zero_grad()
            outputs = lstm_model(X.float())
            loss = criterion(outputs, Y.float())
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                correct = 0
                total = 0
                all_labels = []
                all_probs = []

                lstm_model.eval()
                with torch.inference_mode():
                    for X, Y in test_loader:
                        X, Y = X.to(device), Y.to(device)
                        outputs = lstm_model(X.float())
                        probs = outputs.detach().cpu().numpy().flatten()
                        predicted = (outputs > detect_threshold).float()
                        total += Y.size(0)
                        correct += (predicted == Y.view(-1, 1)).sum().item()
                        all_labels.extend(Y.cpu().numpy())
                        all_probs.extend(probs)

                    accuracy = 100 * correct / total
                    auc_roc = roc_auc_score(all_labels, all_probs)
                    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
                    auc_pr = auc(recall, precision)
                    print(
                        "Iteration: {}. Loss: {}. Accuracy: {}. AUC-ROC: {:.4f}. AUC-PR: {:.4f}".format(
                            iter, loss.item(), accuracy, auc_roc, auc_pr
                        )
                    )

    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = lstm_model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    # Store as onnx for compatibility with Unity Barracuda
    onnx_program = torch.onnx.dynamo_export(lstm_model, torch.randn(1, input_dim))
    onnx_program.save(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx")

    print("\n--- Model Training Complete ---")
    print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)

if __name__ == "__main__":
    main()