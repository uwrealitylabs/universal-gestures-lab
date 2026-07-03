"""
data processing for multi-class dynamic gestures
takes gesture recordings and creates sequences for training
"""

import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data(Dynamic)")
OUTPUT_TRAIN = os.path.join(SCRIPT_DIR, "train_data", "train_sequences_multiclass.pt")
OUTPUT_TEST = os.path.join(SCRIPT_DIR, "test_data", "test_sequences_multiclass.pt")
NORM_STATS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "trained_model", "normalization_stats_multiclass.json")

# map folder names to class IDs
# note: "pos" and "neg" are legacy folder names from binary classification
# class 0 should be "no gesture" to have a base case
GESTURE_CLASSES = {
    "neg": 0,                    # no gesture / resting hand (folder: neg/)
    "pos": 1,                    # fire finger gun (folder: pos/ - legacy name from binary classification)
    "Squeez_Palm_Up": 2,         # squeeze palm up (folder: Squeez_Palm_Up/)
    # "future_gesture": 3,       # add more here when you record new gestures
}

def analyze_sequence_patterns(data_dir):
    """figure out most common sequence length by looking at all the data"""
    sequence_lengths = []

    for gesture_folder in os.listdir(data_dir):
        gesture_path = os.path.join(data_dir, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        # some gestures have pos/neg subfolders, others don't
        if os.path.exists(os.path.join(gesture_path, "pos")):
            for subdir in ['pos', 'neg']:
                subdir_path = os.path.join(gesture_path, subdir)
                if os.path.exists(subdir_path):
                    for json_file in os.listdir(subdir_path):
                        if json_file.endswith('.json'):
                            file_path = os.path.join(subdir_path, json_file)
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            sequences = [data] if isinstance(data, dict) else data
                            for seq in sequences:
                                if 'sequenceData' in seq:
                                    sequence_lengths.append(len(seq['sequenceData']))
        else:
            # no subfolders, JSON files directly in gesture folder
            for json_file in os.listdir(gesture_path):
                if json_file.endswith('.json'):
                    file_path = os.path.join(gesture_path, json_file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    sequences = [data] if isinstance(data, dict) else data
                    for seq in sequences:
                        if 'sequenceData' in seq:
                            sequence_lengths.append(len(seq['sequenceData']))

    if not sequence_lengths:
        return None, None

    # find the most common length
    length_counter = Counter(sequence_lengths)
    mode_length, mode_count = length_counter.most_common(1)[0]
    mode_frequency = mode_count / len(sequence_lengths)

    return mode_length, mode_frequency

def load_gesture_data(data_dir, gesture_name, class_label, window_size, step_size):
    """load all recordings for one gesture type and create sliding windows"""
    sequences = []
    labels = []
    lengths = []

    gesture_path = os.path.join(data_dir, gesture_name)

    if not os.path.exists(gesture_path):
        print(f"  [WARN] Gesture folder not found: {gesture_name}")
        return sequences, labels, lengths

    # check folder structure
    pos_path = os.path.join(gesture_path, "pos")

    if os.path.exists(pos_path):
        # load positive samples from pos/ folder
        json_files = [f for f in os.listdir(pos_path) if f.endswith('.json')]
        print(f"  Found {len(json_files)} files in {gesture_name}/pos/")

        for json_file in json_files:
            file_path = os.path.join(pos_path, json_file)

            with open(file_path, 'r') as f:
                data = json.load(f)

            # sometimes it's a single dict, sometimes a list
            file_sequences = [data] if isinstance(data, dict) else data

            for seq in file_sequences:
                if 'sequenceData' not in seq:
                    continue

                seq_data = np.array(seq['sequenceData'], dtype=np.float32)

                # sliding window with overlap to get more samples
                for start_idx in range(0, len(seq_data) - window_size + 1, step_size):
                    end_idx = start_idx + window_size
                    window = seq_data[start_idx:end_idx]

                    if len(window) == window_size:
                        sequences.append(window)
                        labels.append(class_label)
                        lengths.append(window_size)

    else:
        # no pos/neg structure, just load all files
        json_files = [f for f in os.listdir(gesture_path) if f.endswith('.json')]
        print(f"  Found {len(json_files)} files in {gesture_name}/")

        for json_file in json_files:
            file_path = os.path.join(gesture_path, json_file)

            with open(file_path, 'r') as f:
                data = json.load(f)

            file_sequences = [data] if isinstance(data, dict) else data

            for seq in file_sequences:
                if 'sequenceData' not in seq:
                    continue

                seq_data = np.array(seq['sequenceData'], dtype=np.float32)

                for start_idx in range(0, len(seq_data) - window_size + 1, step_size):
                    end_idx = start_idx + window_size
                    window = seq_data[start_idx:end_idx]

                    if len(window) == window_size:
                        sequences.append(window)
                        labels.append(class_label)
                        lengths.append(window_size)

    return sequences, labels, lengths

def main():
    print("="*60)
    print("Processing Dynamic Gesture Data for Multi-Class Classification")
    print("="*60)

    # figure out window size from the data
    print("\nAnalyzing sequence patterns...")
    mode_length, mode_frequency = analyze_sequence_patterns(DATA_DIR)

    if mode_length is None:
        print("[WARN] No sequences found, using defaults")
        window_size, step_size = 15, 8
    else:
        window_size = mode_length
        step_size = max(1, int(window_size * 0.5))  # 50% overlap for more samples
        print(f"  Window size: {window_size} (appears in {mode_frequency:.1%} of sequences)")
        print(f"  Step size: {step_size} (50% overlap)")

    # load all gesture data
    print(f"\nLoading gesture data...")
    all_sequences = []
    all_labels = []
    all_lengths = []

    available_classes = 0
    for gesture_name, class_label in GESTURE_CLASSES.items():
        print(f"\nClass {class_label}: {gesture_name}")
        sequences, labels, lengths = load_gesture_data(
            DATA_DIR, gesture_name, class_label, window_size, step_size
        )

        if len(sequences) > 0:
            all_sequences.extend(sequences)
            all_labels.extend(labels)
            all_lengths.extend(lengths)
            available_classes += 1
            print(f"  Loaded {len(sequences)} windows")
        else:
            print(f"  No data loaded for this class")

    if len(all_sequences) == 0:
        print("\n[ERROR] No data loaded! Please check your data directory and GESTURE_CLASSES mapping.")
        return

    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Total windows: {len(all_sequences)}")
    print(f"Active classes: {available_classes}")

    label_counter = Counter(all_labels)
    print(f"\nClass distribution:")
    for class_label in sorted(label_counter.keys()):
        gesture_name = [k for k, v in GESTURE_CLASSES.items() if v == class_label][0]
        count = label_counter[class_label]
        print(f"  Class {class_label} ({gesture_name}): {count} samples")

    # convert to numpy
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    lengths = np.array(all_lengths, dtype=np.int32)

    print(f"\nData shape: {X.shape}")
    print(f"Expected: (num_windows, window_size={window_size}, features=17)")

    # compute mean/std for normalization
    print(f"\nComputing normalization statistics...")
    X_flat = X.reshape(-1, X.shape[-1])  # flatten all frames together
    mean = X_flat.mean(axis=0).tolist()
    std = X_flat.std(axis=0).tolist()

    # don't divide by zero if a feature has no variance
    std = [s if s > 1e-6 else 1.0 for s in std]

    print(f"  Mean: {[f'{m:.4f}' for m in mean[:5]]}... (showing first 5)")
    print(f"  Std: {[f'{s:.4f}' for s in std[:5]]}... (showing first 5)")

    # save stats so they can be embedded in the model
    os.makedirs(os.path.dirname(NORM_STATS_PATH), exist_ok=True)
    with open(NORM_STATS_PATH, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    print(f"  Saved to: {NORM_STATS_PATH}")

    # train/test split with stratify to keep classes balanced
    print(f"\nSplitting into train/test (80/20)...")
    X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
        X, y, lengths, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    # pack into dict format
    train_data = {
        'data': torch.tensor(X_train, dtype=torch.float32),
        'labels': torch.tensor(y_train, dtype=torch.long),
        'lengths': torch.tensor(lengths_train, dtype=torch.long)
    }

    test_data = {
        'data': torch.tensor(X_test, dtype=torch.float32),
        'labels': torch.tensor(y_test, dtype=torch.long),
        'lengths': torch.tensor(lengths_test, dtype=torch.long)
    }

    # save
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_TEST), exist_ok=True)

    torch.save(train_data, OUTPUT_TRAIN)
    torch.save(test_data, OUTPUT_TEST)

    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Train data saved to: {OUTPUT_TRAIN}")
    print(f"Test data saved to: {OUTPUT_TEST}")
    print(f"\nNext step: Run model_dynamic_multiclass.py to train the model")

if __name__ == "__main__":
    main()
