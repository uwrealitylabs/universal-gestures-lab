import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
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
    
def split():
    data_files = os.listdir("data")
    
    print("Loading data files")
    full_dataset = []
    train = []
    for dataset_name in data_files:
        full_dataset.extend(process_data("data/"+dataset_name))

    print("Splitting data into training and testing")
    train_size =  int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train, test= torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print("Processing training data")
    train_tensor = torch.tensor(train)
    torch.save(train_tensor, "train_data/train_0.pt")
    
    print("Processing testing data")
    test_tensor = torch.tensor(test)
    torch.save(test_tensor, "test_data/test_0.pt")
    
def main():
    split()
    # process_data(data_path, output_path)

if __name__ == "__main__":
    main()
