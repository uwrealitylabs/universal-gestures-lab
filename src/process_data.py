import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
import os
import json

# WILL NEED TO SOMEHOW MODIFY THE DATA TO SUPPORT > 2 CLASSES

def process_data(dataset_file):
    with open(dataset_file) as f:
        json_data = json.load(f)

    data_list = []
    for entry in json_data:
        hand_data = entry['handData']  
        confidence = entry['confidence']  
        combined_data = hand_data + [confidence]
        data_list.append(combined_data)

    # Convert the list of lists to a torch tensor
    return data_list

def split(dire = "src/data"):
    data_files = os.listdir(dire)

    # split by file first instead of individual frame, as frames from the same recording look identical
    # and splitting after would make near-duplicate frames leak across training and testing
    print("Splitting recording files into training and testing")
    train_files, test_files = train_test_split(data_files, test_size=0.2, random_state=42)

    print("Processing training data")
    train = []
    for dataset_name in train_files:
        train.extend(process_data(dire + "/" + dataset_name))
    train_tensor = torch.tensor(train)
    torch.save(train_tensor, "src/train_data/train_0.pt")

    print("Processing testing data")
    test = []
    for dataset_name in test_files:
        test.extend(process_data(dire + "/" + dataset_name))
    test_tensor = torch.tensor(test)
    torch.save(test_tensor, "src/test_data/test_0.pt")

def main():
    split()
    # process_data(data_path, output_path)

if __name__ == "__main__":
    main()