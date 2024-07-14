import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import json

def process_data(dataset_name, output_path):
    json_data = json.load(open(f"data/{dataset_name}"))
    
    
    
def split():
    full_dataset = os.listdir("data")
    train_size =  int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    print("Processing training data")
    for dataset_name in train_dataset:
        process_data(dataset_name, "train_data")
    
    print("Processing testing data")
    for dataset_name in test_dataset:
        process_data(dataset_name, "test_data")
    
def main():
    split()
    # process_data(data_path, output_path)

if __name__ == "__main__":
    main()
