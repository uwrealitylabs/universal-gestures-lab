"""
Process static hand gesture data into temporal sequences for dynamic gesture recognition.
Creates sequence windows from consecutive frames to train LSTM models for multiclass classification.
"""

import json
import os
import numpy as np
import torch
from collections import defaultdict
import random

# Configuration
SEQUENCE_LENGTH = 20  # Number of frames per sequence window
STEP_SIZE = 5  # Number of frames to skip between sequence windows (for overlap)
TRAIN_SPLIT = 0.6

# Class definitions matching modify_data.py
GESTURE_CLASSES = {
    0: ["closedFistPositive.json"],
    1: ["fingerGunPositive.json"],
    2: ["peaceSignPositive.json"],
    3: ["thumbsupJustin.json", "thumbsuplily.json", "thumbsupNathan.json"],
}

DATA_FOLDER = "src/data"
OUTPUT_TRAIN_PATH = "train_data/train_sequences_0.pt"
OUTPUT_TEST_PATH = "test_data/test_sequences_0.pt"

# Create output directories
os.makedirs("train_data", exist_ok=True)
os.makedirs("test_data", exist_ok=True)


def load_gesture_data(class_idx, gesture_files):
    """Load all data for a specific gesture class."""
    all_data = []

    for filename in gesture_files:
        filepath = os.path.join(DATA_FOLDER, filename)

        if not os.path.exists(filepath):
            print(f"  Warning: File {filepath} not found, skipping...")
            continue

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

                # Extract hand data from each frame
                for frame in data:
                    if isinstance(frame, dict) and "handData" in frame:
                        hand_data = frame["handData"]
                        if len(hand_data) == 17:  # Ensure we have 17 features
                            all_data.append(hand_data)

                print(f"  Loaded {len(data)} frames from {filename}")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue

    return all_data, class_idx


def create_sequences(frames, class_idx, sequence_length=SEQUENCE_LENGTH, step_size=STEP_SIZE):
    """
    Create overlapping sequences from a list of frames.

    Args:
        frames: List of frame data (each frame is a list of 17 features)
        class_idx: Class index (0-3) for this gesture
        sequence_length: Number of frames per sequence
        step_size: Number of frames to skip between sequences

    Returns:
        List of sequences, where each sequence is a 3D array:
        [sequence_length, 18] (17 features + 1 label at the end)
    """
    sequences = []

    if len(frames) < sequence_length:
        print(f"    Warning: Not enough frames ({len(frames)}) for sequence of length {sequence_length}")
        return sequences

    # Create sliding windows
    for start_idx in range(0, len(frames) - sequence_length + 1, step_size):
        end_idx = start_idx + sequence_length
        window = frames[start_idx:end_idx]

        # Create sequence: [sequence_length, 18]
        # Last column is the class label (repeated for all frames in sequence)
        sequence = []
        for frame in window:
            frame_with_label = frame + [class_idx]
            sequence.append(frame_with_label)

        sequences.append(sequence)

    return sequences


def main():
    print("=== Processing Dynamic Gesture Data (Multiclass) ===\n")

    all_sequences = []

    # Process each gesture class
    for class_idx in sorted(GESTURE_CLASSES.keys()):
        gesture_files = GESTURE_CLASSES[class_idx]
        print(f"Processing Class {class_idx} ({gesture_files}):")

        # Load all frames for this class
        frames, _ = load_gesture_data(class_idx, gesture_files)

        if len(frames) == 0:
            print(f"  Error: No data loaded for class {class_idx}")
            continue

        print(f"  Total frames loaded: {len(frames)}")

        # Shuffle frames to mix data from different sources
        random.shuffle(frames)

        # Create sequences
        sequences = create_sequences(frames, class_idx)
        print(f"  Created {len(sequences)} sequences")

        all_sequences.extend(sequences)
        print()

    if len(all_sequences) == 0:
        print("ERROR: No sequences created. Check your data files.")
        return

    print(f"Total sequences created: {len(all_sequences)}")

    # Convert to numpy array for consistency
    # Shape: [num_sequences, sequence_length, 18] (17 features + 1 label)
    all_sequences_np = np.array(all_sequences, dtype=np.float32)
    print(f"Sequences shape: {all_sequences_np.shape}")

    # Shuffle all sequences
    random.shuffle(all_sequences_np)

    # Split into train/test
    num_sequences = len(all_sequences_np)
    train_size = int(num_sequences * TRAIN_SPLIT)

    train_sequences = all_sequences_np[:train_size]
    test_sequences = all_sequences_np[train_size:]

    print(f"\nTrain sequences: {len(train_sequences)} ({len(train_sequences)/num_sequences*100:.1f}%)")
    print(f"Test sequences: {len(test_sequences)} ({len(test_sequences)/num_sequences*100:.1f}%)")

    # Verify class distribution
    print("\nClass distribution in training data:")
    train_labels = train_sequences[:, -1, -1]  # Get labels from last frame of each sequence
    for class_idx in range(4):
        count = np.sum(train_labels == class_idx)
        percentage = count / len(train_labels) * 100
        print(f"  Class {class_idx}: {count} sequences ({percentage:.1f}%)")

    print("\nClass distribution in test data:")
    test_labels = test_sequences[:, -1, -1]
    for class_idx in range(4):
        count = np.sum(test_labels == class_idx)
        percentage = count / len(test_labels) * 100
        print(f"  Class {class_idx}: {count} sequences ({percentage:.1f}%)")

    # Convert to torch tensors
    train_tensor = torch.from_numpy(train_sequences)
    test_tensor = torch.from_numpy(test_sequences)

    # Save to disk
    torch.save(train_tensor, OUTPUT_TRAIN_PATH)
    torch.save(test_tensor, OUTPUT_TEST_PATH)

    print(f"\nSequence data saved:")
    print(f"  Training: {OUTPUT_TRAIN_PATH} (shape: {train_tensor.shape})")
    print(f"  Testing: {OUTPUT_TEST_PATH} (shape: {test_tensor.shape})")
    print("\nDone!")


if __name__ == "__main__":
    main()
