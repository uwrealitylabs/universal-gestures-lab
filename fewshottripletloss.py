import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from confusion_matrix import compute_confusion_matrix

# Model and hyperparameters
input_dim = 17  # number of features for the gesture data
embedding_dim = 10  # embedding size for each input vector
batch_size = 32
num_epochs = 10
threshold = 0.5

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "siamesetriplet_model_weights.json"

# Custom Siamese Network for Few-shot Learning
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def forward_one(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Contrastive Loss for training

# Triplet Loss for training
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        loss = torch.mean(F.relu(positive_distance - negative_distance + self.margin))
        return loss

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Custom dataset that returns pairs of samples for Siamese Network
class GestureDataset(Dataset):
    def __init__(self, data):
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        self.update_pairs()  # Initialize self.pairs during instantiation

    def create_pairs(self):
        # Generate balanced pairs: an equal number of positive and negative pairs
        pairs = []
        positives = self.X[self.Y == 1]
        negatives = self.X[self.Y == 0]
        num_pairs = min(len(positives), len(negatives))

        # Create positive pairs (same label)
        for i in range(num_pairs):
            pairs.append((positives[i], positives[(i + 1) % num_pairs], 1))

        # Create negative pairs (different labels)
        for i in range(num_pairs):
            pairs.append((positives[i], negatives[i], 0))

        np.random.shuffle(pairs)  # Shuffle pairs to mix positives and negatives
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2, label = self.pairs[idx]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def update_pairs(self):
        self.pairs = self.create_pairs()

# Load data and create dataset
def load_data(path):
    data = torch.load(path)
    thumbs_up_data = data[data[:, -1] == 1]
    other_data = data[data[:, -1] == 0]
    combined_data = torch.cat((thumbs_up_data, other_data), dim=0)
    return GestureDataset(combined_data)

# Loaders for training and testing
train_path = "train_data/train_0.pt"
test_path = "test_data/test_0.pt"
train_dataset = load_data(train_path)
test_dataset = load_data(test_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SiameseNetwork(input_dim, embedding_dim)
criterion = TripletLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation
for epoch in range(num_epochs): # Regenerate balanced pairs at each epoch
    model.train()
    for batch_idx, (x1, x2, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output1, output2 = model(x1, x2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

    # Evaluate model after each epoch
    model.eval()
    with torch.no_grad():
        # Testing Metrics
        test_scores, test_labels = [], []
        for x1, x2, label in test_loader:
            output1, output2 = model(x1, x2)
            similarity_scores = F.cosine_similarity(output1, output2)
            test_scores.extend(similarity_scores.cpu().numpy())
            test_labels.extend(label.cpu().numpy())

        test_preds = [1 if score > threshold else 0 for score in test_scores]
        test_accuracy = (accuracy_score(test_labels, test_preds)*100)
        test_auc_roc = roc_auc_score(test_labels, test_scores)
        test_precision = precision_score(test_labels, test_preds)
        test_recall = recall_score(test_labels, test_preds)
        cm = compute_confusion_matrix(test_labels, test_preds)
        print(
            "Epoch [{} / {}], Test Accuracy: {:.4f}, AUC-ROC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Confusion_Matrix: {}".format(
                epoch + 1, num_epochs, test_accuracy, test_auc_roc, test_precision, test_recall,
                ", ".join(f"{key}: {value}" for key, value in cm.items())
            )
        )  

# Extract the model's state dictionary, convert to JSON serializable format
state_dict = model.state_dict()
serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

# Store state dictionary
with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
    json.dump(serializable_state_dict, f)

print("\n--- Model Training Complete ---")
print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)
