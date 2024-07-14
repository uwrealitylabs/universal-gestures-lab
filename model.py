import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :-1]
        label = self.data[idx, -1]
        return sample, label


def load_data(train_path, test_path, batch_size=64):
    # Load the tensor data
    train_tensor = torch.load(train_path)
    test_tensor = torch.load(test_path)

    # Create custom datasets
    train_dataset = CustomDataset(train_tensor)
    test_dataset = CustomDataset(test_tensor)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main():
    train_path = "train_data/thumbsupJustin.pt"
    test_path = "test_data/thumbsupJustin.pt"

    train_loader, test_loader = load_data(train_path, test_path)

    for samples, labels in train_loader:
        print("Samples:", samples)
        print("Labels:", labels)
        break  # Print only the first batch


if __name__ == "__main__":
    main()
