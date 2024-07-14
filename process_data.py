import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import json

def process_data(dataset_file, output_path):
    with open(dataset_file) as f:
        json_data = json.load(f)

    data_list = []
    for entry in json_data:
        hand_data = entry['handData']  # Extract the 'handData' field
        confidence = entry['confidence']  # Extract the 'confidence' field
        combined_data = hand_data + [confidence]
        data_list.append(combined_data)

    # Convert the list of lists to a torch tensor
    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    torch.save(data_tensor, output_path+"/"+dataset_file.split("/")[-1].split(".")[0]+".pt")
    
def split():
    full_dataset = os.listdir("data")
    train_size =  int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    print("Processing training data")
    for dataset_name in train_dataset:
        process_data("data/"+dataset_name, "train_data")
    
    print("Processing testing data")
    for dataset_name in test_dataset:
        process_data("data/"+dataset_name, "test_data")
    
def main():
    split()
    # process_data(data_path, output_path)

if __name__ == "__main__":
    main()
