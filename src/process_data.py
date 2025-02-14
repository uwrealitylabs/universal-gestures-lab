import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import json

# WILL NEED TO SOMEHOW MODIFY THE DATA TO SUPPORT > 2 CLASSES

def process_data(dataset_file, desired_confidence=None):
    with open(dataset_file) as f:
        json_data = json.load(f)

    data_list = []
    for entry in json_data:
        hand_data = entry['handData']  
        confidence = entry['confidence'] if desired_confidence is None else desired_confidence
        combined_data = hand_data + [confidence]
        data_list.append(combined_data)

    # Convert the list of lists to a torch tensor
    return data_list
    
def split(positive_data, data_dir="src/data_with_transform"):
    print(f"\n\n\nSplitting data for {positive_data}")
    print("--------------------------------")
    data_files = os.listdir(data_dir)
    
    print("Loading data files")
    full_dataset = []
    train = []
    print("--------------------------------")
    for dataset_name in data_files:
        print(f"Processing {"positive" if positive_data in dataset_name else "negative"} data : {dataset_name}")
        desired_confidence = 1 if positive_data in dataset_name else 0
        full_dataset.extend(process_data(data_dir + "/" +dataset_name, desired_confidence=desired_confidence))
    print(f"Total data: {len(full_dataset)}")
    print("--------------------------------")

    print("Splitting data into training and testing")
    train_size =  int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train, test= torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print("Processing training data")
    train_tensor = torch.tensor(train)
    torch.save(train_tensor, "src/train_data/train_" + positive_data + ".pt")
    
    print("Processing testing data")
    test_tensor = torch.tensor(test)
    torch.save(test_tensor, "src/test_data/test_" + positive_data + ".pt")
    print("--------------------------------")
    print(f"Finished splitting data for {positive_data}")
    print("--------------------------------")
    
def main():
    for gesture in [
        "erm_actually",
        "finger_gun",
        "fist_up",
        "paper",
        "peace_sign",
        "rock",
        "scissors",
        "thumbs_up",
        "thumbs_down",
        "wave",
    ]:
        split(positive_data=gesture, data_dir="src/data_with_transform")
    # process_data(data_path, output_path)

if __name__ == "__main__":
    main()
