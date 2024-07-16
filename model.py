import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

output_dim = 1 # binary classification for thumbs up or down
input_dim = 17 # 17 features

# Model
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

# Data set
def split_feature_label(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X, self.Y = split_feature_label(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :-1]
        label = self.data[idx, -1]
        return sample, label

# Loader fn
def load_data(dataset, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main():
    train_path = "train_data/train_0.pt"
    test_path = "test_data/test_0.pt"
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    print(train_data[2134])
    train_loader = load_data(train_data)
    test_loader = load_data(test_data)
    
    batch_size = 64
    n_iters = len(train_loader) * 64 * 5 # 5 epochs
    num_epochs = n_iters / (len(train_data)/batch_size)
    
    

if __name__ == "__main__":
    main()
