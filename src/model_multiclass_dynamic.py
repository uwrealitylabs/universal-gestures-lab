import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

TARGET_FRAME_SIZE = 255
TARGET_SEQ_LEN = 32
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-3

CLASS_DIRS = {
    "fire_finger_gun": Path("src/data(Dynamic)/pos"),
    "make_first_palm_up": Path("src/data(Dynamic)/Squeez_Palm_Up/pos"),
}

def load_dynamic_sample(file_path: Path) -> np.ndarray:
    with open(file_path, "r") as f:
        data = json.load(f)

    frames = []

    for frame in data:
        seq = frame["sequenceData"]

        flat_parts = []
        for item in seq:
            arr = np.array(item, dtype=np.float32).flatten()
            flat_parts.append(arr)

        flat = np.concatenate(flat_parts)

        if len(flat) < TARGET_FRAME_SIZE:
            pad = np.zeros(TARGET_FRAME_SIZE - len(flat), dtype=np.float32)
            flat = np.concatenate([flat, pad])
        elif len(flat) > TARGET_FRAME_SIZE:
            flat = flat[:TARGET_FRAME_SIZE]

        frames.append(flat)

    sample = np.array(frames, dtype=np.float32)

    if len(sample) < TARGET_SEQ_LEN:
        pad = np.zeros((TARGET_SEQ_LEN - len(sample), TARGET_FRAME_SIZE), dtype=np.float32)
        sample = np.vstack([sample, pad])
    elif len(sample) > TARGET_SEQ_LEN:
        sample = sample[:TARGET_SEQ_LEN]

    return sample

def load_dataset():
    samples = []
    labels = []
    class_names = list(CLASS_DIRS.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name, class_dir in CLASS_DIRS.items():
        json_files = list(class_dir.glob("*.json"))
        print(f"{class_name}: found {len(json_files)} files")

        for json_file in json_files:
            sample = load_dynamic_sample(json_file)
            samples.append(sample)
            labels.append(class_to_idx[class_name])

    X = np.array(samples, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names

class DynamicGestureClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # -> (batch, input_dim, seq_len)
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    X, y, class_names = load_dataset()

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Class names:", class_names)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DynamicGestureClassifier(input_dim=TARGET_FRAME_SIZE, num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")

    # test one sample
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor[:1])
        probs = torch.softmax(logits, dim=1)
        print("Sample probabilities:", probs)
        print("Predicted class:", class_names[torch.argmax(probs, dim=1).item()])

if __name__ == "__main__":
    main()


# import json
# from pathlib import Path
# import numpy as np

# TARGET_FRAME_SIZE = 255

# def load_dynamic_sample(file_path: Path) -> np.ndarray:
#     with open(file_path, "r") as f:
#         data = json.load(f)

#     frames = []

#     for frame in data:
#         seq = frame["sequenceData"]

#         flat_parts = []
#         for item in seq:
#             arr = np.array(item, dtype=np.float32).flatten()
#             flat_parts.append(arr)

#         flat = np.concatenate(flat_parts)

#         # pad short frames with zeros
#         if len(flat) < TARGET_FRAME_SIZE:
#             pad = np.zeros(TARGET_FRAME_SIZE - len(flat), dtype=np.float32)
#             flat = np.concatenate([flat, pad])

#         # truncate long frames just in case
#         elif len(flat) > TARGET_FRAME_SIZE:
#             flat = flat[:TARGET_FRAME_SIZE]

#         frames.append(flat)

#     return np.array(frames, dtype=np.float32)

# if __name__ == "__main__":
#     sample_path = next(Path("src/data(Dynamic)/pos").glob("*.json"))
#     sample = load_dynamic_sample(sample_path)

#     print("Sample shape:", sample.shape)
#     print("First frame length:", len(sample[0]))
#     print("Last frame length:", len(sample[-1]))
#     print("First frame first 10 values:", sample[0][:10])