import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import os
from . import utils
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import random

# Network aims to retrieve the N-dim fingerprint right before the model outputs to logits

#Hyperparameters
input_dim = 17 # assume same feature # as model.py for now
output_dim = 1 # binary classification (is feature or is not feature)
activation_threshold = 0.7 # threshold for classification as a feature

#Filepaths
SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "gestures_model_weights.json"
source_data_positives = "src/data/closedFistPositive.json" #Select closedFist gesture as training example
source_data_negatives = "src/data/closedFistNegative.json"


class feed_forward_gesture_mlp(nn.Module):
    def __init__(self, in_dim=input_dim, hidden_dim=128, out_dim=output_dim):
        super(feed_forward_gesture_mlp, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim) #input layer -> 128 dim hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2) # 128 dim -> 64 dim hidden layer
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim) # 64 dim layer -> 128 layer 
        self.output = nn.Linear(hidden_dim, out_dim) # final fc layer 128 -> 1 output dim 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return torch.sigmoid(x)


# Data set
def split_feature_label(data):
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y #X is features, Y is positive/negative label (1/0)

class CustomDataset(torch.utils.data.Dataset):
    # Custom class inherits from torch Dataset to handle loading of JSON data
    def __init__(self, data):
        (self.X, self.Y) = split_feature_label(data)
    def __len__(self):
        return len(self.X) #length of the data input data is equal to number of features
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def load_data(dataset, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def convert_to_tensors(data: object) -> torch.Tensor:
    # Converts JSON object into tensors storing featurs and labels
    all_X = [item["handData"] for item in data] 
    all_Y = [item["label"] for item in data]
    X_tensor = torch.tensor(all_X, dtype=torch.float32)
    Y_tensor = torch.tensor(all_Y, dtype=torch.float32)
    return (X_tensor, Y_tensor)


def main():
    with open(source_data_positives, 'r') as f:
        positive_examples = json.load(f)
    with open(source_data_negatives, 'r') as f:
        negative_examples = json.load(f)
    
    combined_data = positive_examples + negative_examples
    random.shuffle(combined_data)  # Shuffle the combined data
    (train_data_JSON, test_data_JSON) = train_test_split(combined_data, test_size=0.2, random_state=42)

    train_tensors = convert_to_tensors(train_data_JSON)
    test_tensors = convert_to_tensors(test_data_JSON)

    # Unpack tensors and 
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(*test_tensors), batch_size=64, shuffle=False)

   # Initialize model
    model = feed_forward_gesture_mlp()
   # ...