import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Model and hyperparameters
input_dim = 17  # features per timestep
hidden_dim = 32  # LSTM hidden size
embedding_dim = 10  # final embedding size
batch_size = 32
num_epochs = 20
margin = 1.0  # how much farther apart negatives need to be than positives before loss = 0
val_split = 0.15  # held-out fraction for per-epoch validation
threshold = 0.5  # default cosine-similarity cutoff for "same gesture" (gets overridden later using val data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paths built off this file's location so it doesn't matter where you run it from -
# the .pt files live at the repo root, not under src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAIN_PATH = os.path.join(SCRIPT_DIR, "train_data", "train_sequences_multiclass.pt")
TEST_PATH = os.path.join(SCRIPT_DIR, "test_data", "test_sequences_multiclass.pt")
SAVE_MODEL_PATH = os.path.join(REPO_ROOT, "trained_model")
SAVE_MODEL_FILENAME = "siamesetriplet_model_weights.json"


def split_feature_label(data):
    # train_sequences_multiclass.pt stores data as a dict, not a flat tensor -
    # features and labels are already separate, no slicing needed
    X = data['data']
    Y = data['labels']
    return X, Y


class SiameseNetwork(nn.Module):
    """Encodes a (seq_len, input_dim) gesture window into a single embedding via LSTM."""

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward_one(self, x):
        # x is (batch, seq_len, input_dim), one gesture window. Run the LSTM
        # over the whole thing and just keep the final hidden state: that's
        # one vector summarizing the whole motion, then project to embedding_dim.
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

    def forward(self, anchor, positive, negative):
        # same weights for all three, one shared
        # encoder instead of three separate ones
        return self.forward_one(anchor), self.forward_one(positive), self.forward_one(negative)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # anchor should be closer to positive than to negative by at least
        # `margin`. relu() zeroes the loss once that's already satisfied, so
        # we're only penalizing triplets the model hasn't figured out yet
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        return torch.mean(F.relu(positive_distance - negative_distance + self.margin))


class TripletGestureDataset(Dataset):
    """Randomly samples (anchor, positive, negative) triplets across all classes. Call resample() each epoch."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y.long()
        self.by_label = {}
        for label in torch.unique(self.Y).tolist():
            self.by_label[label] = torch.nonzero(self.Y == label, as_tuple=True)[0].tolist()
        self.resample()

    def resample(self):
        # every sample takes a turn as the anchor - grab a random same-label
        # sample as the positive and a random different-label sample as the
        # negative. random, not sequential, so we're not stuck training on
        # the same handful of triplets every epoch
        triplets = []
        for anchor_idx in range(len(self.X)):
            anchor_label = self.Y[anchor_idx].item()
            same_class = self.by_label[anchor_label]
            other_labels = [l for l in self.by_label if l != anchor_label]
            if len(same_class) < 2 or not other_labels:
                continue  # nothing to pair this one with, skip it
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


class PairGestureDataset(Dataset):
    """Randomly samples (x1, x2, same_label) pairs for eval metrics (AUC-ROC/precision/recall). Call resample() each epoch."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y.long()
        self.by_label = {}
        for label in torch.unique(self.Y).tolist():
            self.by_label[label] = torch.nonzero(self.Y == label, as_tuple=True)[0].tolist()
        self.resample()

    def resample(self):
        # one same-label pair and one different-label pair per sample, both
        # picked randomly - only used for eval metrics, never training
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


def load_split_data(path):
    # load a .pt file, split into X/Y
    data = torch.load(path, weights_only=False)
    return split_feature_label(data)


def evaluate_pairs(model, loader, threshold=threshold):
    # embed both sides, check cosine similarity (1 = same direction, -1 =
    # opposite), call it "same gesture" if it clears the threshold
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x1, x2, label in loader:
            x1, x2 = x1.to(device), x2.to(device)
            emb1, emb2 = model.forward_one(x1), model.forward_one(x2)
            similarity = F.cosine_similarity(emb1, emb2)
            scores.extend(similarity.cpu().numpy())
            labels.extend(label.numpy())
    preds = [1 if s > threshold else 0 for s in scores]
    return labels, scores, preds


def main():
    # -Load data-
    # test set doesn't get touched again until the very end
    X_train_full, Y_train_full = load_split_data(TRAIN_PATH)
    X_test, Y_test = load_split_data(TEST_PATH)

    # -Carve out a validation split-
    # stratified so each class keeps roughly the same proportion on both
    # sides; this is what lets us watch for overfitting/tune stuff without
    # peeking at test
    indices = np.arange(len(X_train_full))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=Y_train_full.numpy(), random_state=42
    )
    X_train, Y_train = X_train_full[train_idx], Y_train_full[train_idx]
    X_val, Y_val = X_train_full[val_idx], Y_train_full[val_idx]

    # -Datasets/loaders-
    # training runs on triplets (what the loss needs), val/test run on pairs
    # (what the eval metrics need)
    train_dataset = TripletGestureDataset(X_train, Y_train)
    val_dataset = PairGestureDataset(X_val, Y_val)
    test_dataset = PairGestureDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SiameseNetwork(input_dim, hidden_dim, embedding_dim).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_dataset.resample()  # new random triplets each epoch, not just once at the start
        epoch_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()
            a_emb, p_emb, n_emb = model(anchor, positive, negative)
            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # pick the actual classification threshold off the validation ROC curve
    # instead of guessing 0.5:  AUC-ROC being near-perfect doesn't mean 0.5
    # is where the real score distributions cross
    val_dataset.resample()
    val_labels, val_scores, _ = evaluate_pairs(model, val_loader)
    fpr, tpr, roc_thresholds = roc_curve(val_labels, val_scores)
    best_threshold = roc_thresholds[np.argmax(tpr - fpr)]
    print(f"\nSelected threshold from validation ROC curve: {best_threshold:.4f}")

    # final, one-time evaluation on the untouched test set
    test_labels, test_scores, test_preds = evaluate_pairs(model, test_loader, threshold=best_threshold)
    test_acc = accuracy_score(test_labels, test_preds) * 100
    test_auc = roc_auc_score(test_labels, test_scores)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    print(
        f"\nFinal Test - Accuracy: {test_acc:.2f}, AUC-ROC: {test_auc:.4f}, "
        f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}"
    )

    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    with open(os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME), "w") as f:
        json.dump(serializable_state_dict, f)

    print("\n--- Model Training Complete ---")
    print("Model weights saved to", os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME))


if __name__ == "__main__":
    main()
