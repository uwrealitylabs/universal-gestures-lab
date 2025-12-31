"""
Multi-class dynamic gesture recognition using Conv2D
Extended from binary version to handle multiple gesture types
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config for multi-class setup
num_classes = 3  # no gesture, fire finger gun, squeeze palm. can add more later
input_dim = 17   # hand tracking features per frame

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_model")
SAVE_MODEL_FILENAME = "model_dynamic_multiclass_weights.json"
TRAIN_PATH = os.path.join(SCRIPT_DIR, "train_data", "train_sequences_multiclass.pt")
TEST_PATH = os.path.join(SCRIPT_DIR, "test_data", "test_sequences_multiclass.pt")
NORM_STATS_PATH = os.path.join(SAVE_MODEL_PATH, "normalization_stats_multiclass.json")

class FixedNormalization(nn.Module):
    """normalization layer with fixed mean/std - embedded in model so Unity doesn't need to normalize"""
    def __init__(self, mean, std):
        super(FixedNormalization, self).__init__()
        # use register_buffer so these are saved with model but not trained
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        if x.dim() == 4:  # NHWC format from Unity
            mean = self.mean.view(1, 1, 1, -1)
            std = self.std.view(1, 1, 1, -1)
            x = (x - mean) / std
        elif x.dim() == 3:
            x = (x - self.mean) / self.std
        return x

class Conv2D_MultiClass_Model(nn.Module):
    def __init__(self, input_dim, num_classes, sequence_length=15, norm_mean=None, norm_std=None):
        super(Conv2D_MultiClass_Model, self).__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # embed normalization if we have stats
        if norm_mean is not None and norm_std is not None:
            self.normalize = FixedNormalization(norm_mean, norm_std)
        else:
            self.normalize = None

        # Conv2D treats sequence as 2D image - time becomes height dimension
        # kernel (3,1) slides along time axis
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # pool everything to 1x1
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(128, num_classes)  # output logits for each class

        self.relu = nn.ReLU()

    def forward(self, x):
        # expecting NHWC format from Unity: (batch, seq_length, 1, features)

        if x.dim() == 3:
            x = x.unsqueeze(2)  # add width dim if missing

        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D with shape {x.shape}")

        # normalize if we have stats
        if self.normalize is not None:
            x = self.normalize(x)

        # PyTorch Conv2D wants NCHW so permute from NHWC
        x = x.permute(0, 3, 1, 2)

        # conv blocks with batchnorm
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # pool everything down to single values
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # dropout then final classification
        x = self.dropout(x)
        out = self.fc(x)  # raw logits, no softmax (CrossEntropy does that)

        return out

def split_feature_label(data, labels, lengths):
    """separate features from labels, convert labels to long for CrossEntropy"""
    X = data
    Y = labels.long()  # needs to be long not float
    return X, Y, lengths

def main():
    print("=" * 60)
    print("Training Conv2D Multi-Class Model")
    print("=" * 60)

    # load processed data
    print(f"\nLoading data...")
    print(f"  Train: {TRAIN_PATH}")
    print(f"  Test: {TEST_PATH}")

    train_loaded = torch.load(TRAIN_PATH, weights_only=False)
    test_loaded = torch.load(TEST_PATH, weights_only=False)

    train_data, train_labels, train_lengths = train_loaded['data'], train_loaded['labels'], train_loaded['lengths']
    test_data, test_labels, test_lengths = test_loaded['data'], test_loaded['labels'], test_loaded['lengths']

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # load norm stats if we have them
    norm_mean, norm_std = None, None
    if os.path.exists(NORM_STATS_PATH):
        print(f"\nLoading normalization statistics from {NORM_STATS_PATH}")
        with open(NORM_STATS_PATH, 'r') as f:
            norm_stats = json.load(f)
            norm_mean = norm_stats['mean']
            norm_std = norm_stats['std']
        print("[OK] Normalization will be embedded in the model")
    else:
        print("[WARN] No normalization statistics found. Training without normalization.")

    batch_size = 64
    n_iters = len(train_data) * 10  # 10 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))

    X_train, y_train, lengths_train = split_feature_label(train_data, train_labels, train_lengths)
    X_test, y_test, lengths_test = split_feature_label(test_data, test_labels, test_lengths)

    # check how balanced the classes are
    print(f"\nLabel distribution:")
    unique, counts = torch.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples")

    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train, lengths_train)), shuffle=True, batch_size=batch_size
    )

    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test, lengths_test)), shuffle=False, batch_size=batch_size
    )

    # create model - norm stats get embedded if we loaded them
    model = Conv2D_MultiClass_Model(input_dim, num_classes, norm_mean=norm_mean, norm_std=norm_std).to(device)

    print(f"\nModel configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output classes: {num_classes}")
    print(f"  Device: {device}")

    # CrossEntropy for multi-class (handles softmax internally)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining configuration:")
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")

    iter = 0

    # training loop
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (X, Y, lengths) in enumerate(train_loader):
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            optimizer.zero_grad()

            outputs = model(X.float())
            loss = criterion(outputs, Y)  # expects Y as class indices

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # get predicted class
            train_total += Y.size(0)
            train_correct += (predicted == Y).sum().item()

            iter += 1

            # eval every 500 iters
            if iter % 500 == 0:
                model.eval()
                test_correct = 0
                test_total = 0
                all_labels = []
                all_predictions = []

                with torch.inference_mode():
                    for X_test_batch, Y_test_batch, lengths_test_batch in test_loader:
                        X_test_batch, Y_test_batch = X_test_batch.to(device), Y_test_batch.to(device)
                        outputs = model(X_test_batch.float())
                        _, predicted = torch.max(outputs, 1)
                        test_total += Y_test_batch.size(0)
                        test_correct += (predicted == Y_test_batch).sum().item()
                        all_labels.extend(Y_test_batch.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())

                train_accuracy = 100 * train_correct / train_total
                test_accuracy = 100 * test_correct / test_total

                print(f"Iteration {iter}: Loss={loss.item():.4f}, Train Acc={train_accuracy:.1f}%, Test Acc={test_accuracy:.1f}%")

                # check if model is predicting all classes
                from collections import Counter
                label_dist = Counter(all_labels)
                pred_dist = Counter(all_predictions)
                print(f"  Label dist: {dict(label_dist)}")
                print(f"  Pred dist: {dict(pred_dist)}")

                model.train()

    # final eval with full metrics
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}\n")

    model.eval()
    final_labels = []
    final_predictions = []

    with torch.inference_mode():
        for X, Y, lengths in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X.float())
            _, predicted = torch.max(outputs, 1)
            final_labels.extend(Y.cpu().numpy())
            final_predictions.extend(predicted.cpu().numpy())

    # detailed metrics
    class_names = [f"Gesture {i}" for i in range(num_classes)]
    print("Classification Report:")
    print(classification_report(final_labels, final_predictions, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(final_labels, final_predictions))

    final_accuracy = 100 * sum(p == l for p, l in zip(final_predictions, final_labels)) / len(final_labels)
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")

    # save the model
    print(f"\n{'='*60}")
    print("Saving Model")
    print(f"{'='*60}\n")

    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # save as JSON for easy inspection
    with open(os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME), "w") as f:
        json.dump(serializable_state_dict, f)

    # ONNX export for Unity
    dummy_input = torch.randn(1, 15, 1, input_dim).to(device)
    onnx_filename = SAVE_MODEL_FILENAME.replace('.json', '.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(SAVE_MODEL_PATH, onnx_filename),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"[OK] Model weights saved to {SAVE_MODEL_PATH}/{SAVE_MODEL_FILENAME}")
    print(f"[OK] ONNX model saved to {SAVE_MODEL_PATH}/{onnx_filename}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
