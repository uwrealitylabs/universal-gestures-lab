import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import random

# Model and hyperparameters
input_dim = 17  # number of features for the gesture data
embedding_dim = 10  # embedding size for each input vector
batch_size = 32
num_epochs = 10
threshold = 0.5

SAVE_MODEL_PATH = "../trained_model/"
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

# Custom dataset that returns pairs of samples for Siamese Network
class TripletGestureDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y.long()
        self.by_label = {}
        for label in torch.unique(self.Y).tolist():
            self.by_label[label] = torch.nonzero(self.Y == label, as_tuple=True)[0].tolist()
        self.resample()

    def resample(self):
        triplets = []
        for anchor_idx in range(len(self.X)):
            anchor_label = self.Y[anchor_idx].item()
            same_class = self.by_label[anchor_label]
            other_labels = [l for l in self.by_label if l != anchor_label]
            if len(same_class) < 2 or not other_labels:
                continue
            pos_idx = anchor_idx
            while pos_idx == anchor_idx:
                pos_idx = random.choice(same_class)
            neg_label = random.choice(other_labels)
            neg_idx = random.choice(self.by_label[neg_label])
            triplets.append((anchor_idx, pos_idx, neg_idx))
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return self.X[a].float(), self.X[p].float(), self.X[n].float()

def load_data(path):
    data = torch.load(path)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

class PairGestureDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y.long()
        self.by_label = {}
        for label in torch.unique(self.Y).tolist():
            self.by_label[label] = torch.nonzero(self.Y == label, as_tuple=True)[0].tolist()
        self.resample()

    def resample(self):
        pairs = []
        labels = list(self.by_label.keys())
        for idx in range(len(self.X)):
            label = self.Y[idx].item()
            same_class = self.by_label[label]
            if len(same_class) > 1:
                j = idx
                while j == idx:
                    j = random.choice(same_class)
                pairs.append((idx, j, 1))
            other_labels = [l for l in labels if l != label]
            if other_labels:
                neg_label = random.choice(other_labels)
                j = random.choice(self.by_label[neg_label])
                pairs.append((idx, j, 0))
        random.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        return self.X[i].float(), self.X[j].float(), torch.tensor(label, dtype=torch.float32)

train_path = "train_data/train_0.pt"
test_path = "test_data/test_0.pt"
X_train_full, Y_train_full = load_data(train_path)
X_test, Y_test = load_data(test_path)

val_split = 0.15
indices = np.arange(len(X_train_full))
train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=Y_train_full.numpy(), random_state=42)
X_train, Y_train = X_train_full[train_idx], Y_train_full[train_idx]
X_val, Y_val = X_train_full[val_idx], Y_train_full[val_idx]

train_dataset = TripletGestureDataset(X_train, Y_train)
val_dataset = PairGestureDataset(X_val, Y_val)
test_dataset = PairGestureDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SiameseNetwork(input_dim, embedding_dim)
criterion = TripletLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation
for epoch in range(num_epochs):
    model.train()
    train_dataset.resample()  # fresh random triplets each epoch, not just once at startup
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
      optimizer.zero_grad()
      anchor_out = model.forward_one(anchor)
      positive_out = model.forward_one(positive)
      negative_out = model.forward_one(negative)
      loss = criterion(anchor_out, positive_out, negative_out)
      loss.backward()
      optimizer.step()

      # Evaluate on validation data after each epoch - test set stays untouched
    model.eval()
    val_dataset.resample()
    with torch.no_grad():
        val_scores, val_labels = [], []
        for x1, x2, label in val_loader:
            output1, output2 = model(x1, x2)
            similarity_scores = F.cosine_similarity(output1, output2)
            val_scores.extend(similarity_scores.cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    val_preds = [1 if score > threshold else 0 for score in val_scores]
    val_accuracy = (accuracy_score(val_labels, val_preds) * 100)
    val_auc_roc = roc_auc_score(val_labels, val_scores)
    val_precision = precision_score(val_labels, val_preds)
    val_recall = recall_score(val_labels, val_preds)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Val Accuracy: {val_accuracy:.4f}, AUC-ROC: {val_auc_roc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    # Final, one-time evaluation on the untouched test set
model.eval()
with torch.no_grad():
    test_scores, test_labels = [], []
    for x1, x2, label in test_loader:
        output1, output2 = model(x1, x2)
        similarity_scores = F.cosine_similarity(output1, output2)
        test_scores.extend(similarity_scores.cpu().numpy())
        test_labels.extend(label.cpu().numpy())

    test_preds = [1 if score > threshold else 0 for score in test_scores]
    test_accuracy = (accuracy_score(test_labels, test_preds) * 100)
    test_auc_roc = roc_auc_score(test_labels, test_scores)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    print(
        f"\nFinal Test - Accuracy: {test_accuracy:.4f}, AUC-ROC: {test_auc_roc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

# Extract the model's state dictionary, convert to JSON serializable format
state_dict = model.state_dict()
serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

# Store state dictionary
with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
    json.dump(serializable_state_dict, f)

print("\n--- Model Training Complete ---")
print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)
