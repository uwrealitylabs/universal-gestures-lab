import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import warnings # Ignore warning that .onnx export is using ONNX opset 18 

#Hyperparameters
input_dim = 17 # assume same feature # as model.py for now
output_dim = 4 # 4 output classes for gestures (from model_multi_class.py)
activation_threshold = 0.7 # threshold for classification as a feature
learning_rate = 0.001 
epochs = 20 # number of epochs to train model (set to 10 for testing)

#Filepaths
SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "gestures_model_weights.json"

TESTING_DATA_PATH = "src/test_data/"
TRAINING_DATA_PATH = "src/train_data/"


class feed_forward_gesture_mlp(nn.Module):
    def __init__(self, in_dim=input_dim, hidden_dim=128, out_dim=output_dim):
        super(feed_forward_gesture_mlp, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim) #input layer -> 128 dim hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2) # 128 dim -> 64 dim hidden layer
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim) # 64 dim layer -> 128 layer 
        self.output = nn.Linear(hidden_dim, out_dim) # final fc layer 128 -> 4 output dims 
    
    def forward(self, x):
        x_1 = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        n_dim_fingerprint = self.fc3(x_2)
        x_3 = F.relu(n_dim_fingerprint)
        x_4 = self.output(x_3)
        return {"final_logits": x_4, "n_dim_fingerprint": n_dim_fingerprint}

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


def export_to_onnx(model):
    model.eval()
    
     # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Create directory if it does not exist
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    # Store as onnx for compatibility with Unity Barracuda
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch.onnx.dynamo_export only implements opset version 18.*")
        onnx_program = torch.onnx.dynamo_export(model, torch.randn(1, input_dim))
        onnx_program.save(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx")
    print("\nModel weights and .onnx successfully saved to", SAVE_MODEL_PATH)

def load_json_data(test_path, train_path):
    # Load all data files first (without processing)
    testing_files_data = {}
    training_files_data = {}

    print("=== Loading Testing Data Files ===")
    for file in os.listdir(test_path):
        if file.endswith(".json"):
            file_path = os.path.join(test_path, file)
            print(f"Loading file: {file}")
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    testing_files_data[file] = data
                    print(f"  {file}: {len(data)} samples")

    print("\n=== Loading Training Data Files ===")
    for file in os.listdir(train_path):
        if file.endswith(".json"):
            file_path = os.path.join(train_path, file)
            print(f"Loading file: {file}")
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    training_files_data[file] = data
                    print(f"  {file}: {len(data)} samples")
    print("\n=== Data Loading Complete ===")
    return (testing_files_data, training_files_data)
    
    
def get_min_data_length(train_files_data, test_files_data):
     # Find minimum length across all files
    all_lengths = []
    for _ , data in train_files_data.items():
        all_lengths.append(len(data))
    for _ , data in test_files_data.items():
        all_lengths.append(len(data))

    if not all_lengths:
        print("ERROR: No valid data files found.")
        return

    min_length = min(all_lengths)
    print(f"\n=== Data Balancing ===")
    print(f"Minimum file length: {min_length} samples")
    print(f"Will extract {min_length} samples from each file for balanced dataset")
    return min_length


def process_data(min_l, train_files_data, test_files_data):
    testing_data_array = []
    training_data_array = []
    print("\n=== Processing Testing Data ===")
    for file, data in test_files_data.items():
        print(f"Processing {file}: extracting {min_l} from {len(data)} samples")
        # Take only up to min_length samples from each file
        balanced_data = data[:min_l]

        for item in balanced_data:
            if isinstance(item, dict) and "handData" in item and "label" in item:
                # Create flat list: handData + label
                row = item["handData"] + [item["label"]]
                testing_data_array.append(row)
    
    print("\n=== Processing Training Data ===")
    for file, data in train_files_data.items():
        print(f"Processing {file}: extracting {min_l} from {len(data)} samples")
        # Take only up to min_length samples from each file
        balanced_data = data[:min_l]
        for item in balanced_data:
            if isinstance(item, dict) and "handData" in item and "label" in item:
                # Create flat list: handData + label
                row = item["handData"] + [item["label"]]
                training_data_array.append(row)

    if not training_data_array or not testing_data_array:
        print(
            "ERROR: No valid data loaded. Check your JSON file format and run modify_data.py first."
        )
        return
    return (testing_data_array, training_data_array)


def convert_to_tensors(train_data, test_data):
    train_data = np.array(train_data, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.float32)
    
    print(f"\n=== Final Dataset Statistics ===")
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    print(f"Total combined data: {len(train_data) + len(test_data)} samples")
    
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    # output stats
    print(
        f"Training data: {len(train_data)} samples ({len(train_data)/(len(train_data) + len(test_data))*100:.1f}%)"
    )
    print(
        f"Test data: {len(test_data)} samples ({len(test_data)/(len(test_data) + len(train_data))*100:.1f}%)"
    )
    
    return (test_data, train_data) #outputs numpy arrays of test and train data


def main():
    (test_files_data, train_files_data) = load_json_data(TESTING_DATA_PATH, TRAINING_DATA_PATH)
    min_length = get_min_data_length(test_files_data, train_files_data)
    (test_data, train_data) = process_data(min_length, train_files_data, test_files_data)
    (test_data_tensor, train_data_tensor) = convert_to_tensors(train_data, test_data)
    
    os.makedirs("src/train_data", exist_ok=True)
    os.makedirs("src/test_data", exist_ok=True)
    
    # Save the split data
    torch.save(train_data_tensor, "src/train_data/train_split.pt")
    torch.save(test_data_tensor, "src/test_data/test_split.pt")
    
    batch_size = 16  # Keep consistent
    
    X_train = torch.tensor(train_data_tensor[:, :-1], dtype=torch.float32)
    y_train = torch.tensor(train_data_tensor[:, -1], dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=batch_size
    )

    X_test = torch.tensor(test_data_tensor[:, :-1], dtype=torch.float32)
    y_test = torch.tensor(test_data_tensor[:, -1], dtype=torch.long)
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=False, batch_size=batch_size
    )

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Unique labels in training: {torch.unique(y_train)}")


   # Initialize model
    model = feed_forward_gesture_mlp()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Get a prediction
        total_loss = 0.0
        correct = 0
        total = 0
        
        for _, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X)["final_logits"]
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

        # Evaluate accuracy of model
        model.eval() 
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for X, Y in test_loader:
                outputs = model(X)["final_logits"]
                _, predicted = torch.max(outputs, 1)
                test_total += Y.size(0)
                test_correct += (predicted == Y).sum().item()
                all_predictions.extend(predicted.numpy())
                all_labels.extend(Y.numpy())
        
        train_accuracy = 100 * correct / total
        test_accuracy = 100 * test_correct / test_total
        avg_loss = total_loss / len(train_loader)
        
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Test Acc: {test_accuracy:.2f}%")
        model.train()

    #Final Evaluation
    print("\n--- Model Training/Evaluation Complete ---")
    print(f"Final training accuracy: {train_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Total training samples: {total}, Total test samples: {test_total}")
    export_to_onnx(model) # Export the trained model to ONNX format
    
if __name__ == "__main__":
    main()