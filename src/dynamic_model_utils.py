import json
from pathlib import Path
import numpy as np
import torch.nn as nn

TARGET_FRAME_SIZE = 255
TARGET_SEQ_LEN = 32

CLASS_DIRS = {
    "fire_finger_gun": Path("src/data(Dynamic)/pos"),
    "make_first_palm_up": Path("src/data(Dynamic)/Squeez_Palm_Up/pos"),
}

MODEL_SAVE_DIR = Path("trained_model")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS_PATH = MODEL_SAVE_DIR / "dynamic_multiclass_model.pth"
CLASS_NAMES_PATH = MODEL_SAVE_DIR / "dynamic_multiclass_class_names.json"


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
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.classifier(x)
        return x


def save_class_names(class_names):
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)


def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)