import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def analyze_sequence_patterns(data_dir):
    """
    Find the most common sequence length in the data.
    
    Args:
        data_dir: Directory containing pos/ and neg/ subdirectories with JSON files
    
    Returns:
        tuple: (mode_length, mode_frequency) or (None, None) if no data found
    """
    sequence_lengths = []
    
    # Scan both pos and neg directories
    for subdir in ['pos', 'neg']:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            continue
            
        for json_file in os.listdir(subdir_path):
            if not json_file.endswith('.json'):
                continue
                
            file_path = os.path.join(subdir_path, json_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both single dict and list formats
            sequences = [data] if isinstance(data, dict) else data
            
            for seq in sequences:
                if 'sequenceData' in seq:
                    sequence_lengths.append(len(seq['sequenceData']))
    
    if not sequence_lengths:
        return None, None
    
    # Find most common sequence length
    length_counter = Counter(sequence_lengths)
    mode_length, mode_count = length_counter.most_common(1)[0]
    mode_frequency = mode_count / len(sequence_lengths)
    
    return mode_length, mode_frequency

def adaptive_window_parameters(data_dir, overlap_ratio=0.5):
    """
    Determine window parameters based on most common sequence length in data.
    
    Args:
        data_dir: Directory containing the data
        overlap_ratio: Overlap ratio (0.5 = 50% overlap)
    
    Returns:
        tuple: (window_size, step_size)
    """
    mode_length, mode_frequency = analyze_sequence_patterns(data_dir)
    
    if mode_length is None:
        print("Warning: No sequences found, using defaults")
        return 15, 8
    
    # Use most common sequence length as window size
    window_size = mode_length
    step_size = max(1, int(window_size * (1 - overlap_ratio)))
    
    print(f"Detected window size: {window_size} samples (appears in {mode_frequency:.1%} of sequences)")
    print(f"Step size: {step_size} samples ({overlap_ratio*100:.0f}% overlap)")
    
    return window_size, step_size

def load_dynamic_data(data_dir, label, window_size, step_size):
    """
    Load dynamic gesture data from JSON files with overlapping windows.
    
    Args:
        data_dir: Directory containing JSON files
        label: Label for this data (1 for positive, 0 for negative)
        window_size: Size of each window in samples
        step_size: Step size for sliding window in samples
    
    Returns:
        sequences: List of sequences (overlapping windows)
        labels: List of labels
    """
    sequences = []
    labels = []
    
    # List all JSON files in the directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single object and array formats
        if isinstance(data, dict):
            data = [data]
        
        for entry in data:
            if 'sequenceData' in entry and 'confidence' in entry:
                sequence = entry['sequenceData']
                confidence = entry['confidence']
                
                # For positive examples: require confidence=1, for negative examples: accept all (confidence=0)
                if (label == 1 and confidence == 1) or (label == 0):
                    # Create overlapping windows from the sequence
                    seq_len = len(sequence)
                    
                    if seq_len <= window_size:
                        # If sequence is shorter than window, use as is
                        sequences.append(sequence)
                        labels.append(label)
                    else:
                        # Create overlapping windows
                        for start_idx in range(0, seq_len - window_size + 1, step_size):
                            end_idx = start_idx + window_size
                            window = sequence[start_idx:end_idx]
                            sequences.append(window)
                            labels.append(label)
                        
                        # Always include the last window to ensure we don't miss the end
                        if (seq_len - window_size) % step_size != 0:
                            last_window = sequence[-window_size:]
                            sequences.append(last_window)
                            labels.append(label)
    
    return sequences, labels

def pad_sequences(sequences, max_length=None):
    """
    Pad sequences to the same length
    
    Args:
        sequences: List of sequences of varying lengths
        max_length: Maximum sequence length (if None, use longest sequence)
    
    Returns:
        Padded sequences as numpy array
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) > max_length:
            # Truncate if too long
            padded_seq = seq[:max_length]
        else:
            # Pad with zeros if too short
            padded_seq = seq + [[0.0] * len(seq[0])] * (max_length - len(seq))
        padded.append(padded_seq)
    
    return np.array(padded)

def compute_feature_statistics(sequences):
    """
    Compute mean and standard deviation for each feature across all sequences
    
    Args:
        sequences: List of sequences, each sequence is a list of feature vectors
        
    Returns:
        mean: Mean for each feature (shape: [num_features])
        std: Standard deviation for each feature (shape: [num_features])
    """
    # Flatten all sequences into a single array for statistics computation
    all_features = []
    for seq in sequences:
        for timestep in seq:
            all_features.append(timestep)
    
    features_array = np.array(all_features)  # Shape: (total_timesteps, num_features)
    
    # Compute statistics
    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0)
    
    # Avoid division by zero - replace zero std with 1.0
    std = np.where(std == 0, 1.0, std)
    
    return mean, std

def normalize_sequences(sequences, mean, std):
    """
    Normalize sequences using provided mean and standard deviation
    
    Args:
        sequences: List of sequences to normalize
        mean: Mean for each feature
        std: Standard deviation for each feature
        
    Returns:
        normalized_sequences: List of normalized sequences
    """
    normalized = []
    for seq in sequences:
        normalized_seq = []
        for timestep in seq:
            # Normalize each feature: (feature - mean) / std
            normalized_timestep = [(f - m) / s for f, m, s in zip(timestep, mean, std)]
            normalized_seq.append(normalized_timestep)
        normalized.append(normalized_seq)
    
    return normalized

def create_sequence_tensors(positive_dir, negative_dir, test_split=0.2, overlap_ratio=0.5):
    """
    Create training and testing tensors from dynamic gesture data with auto-detected window parameters.
    
    Args:
        positive_dir: Directory containing positive examples
        negative_dir: Directory containing negative examples (optional)
        test_split: Fraction of data to use for testing
        overlap_ratio: Overlap ratio for windows (0.5 = 50% overlap)
    
    Returns:
        train_data: Training data tensor
        train_labels: Training labels tensor
        train_lengths: Training sequence lengths tensor
        test_data: Testing data tensor
        test_labels: Testing labels tensor
        test_lengths: Testing sequence lengths tensor
        mean: Feature means for normalization
        std: Feature standard deviations for normalization
    """
    
    # Auto-detect window parameters from data
    parent_dir = os.path.dirname(positive_dir)
    print(f"\nAnalyzing sequence patterns...")
    window_size, step_size = adaptive_window_parameters(parent_dir, overlap_ratio)
    
    # Load positive examples with overlapping windows
    pos_sequences, pos_labels = load_dynamic_data(positive_dir, label=1, 
                                                 window_size=window_size, 
                                                 step_size=step_size)
    
    all_sequences = pos_sequences
    all_labels = pos_labels
    
    print(f"Loaded {len(pos_sequences)} positive windows")
    
    # Load negative examples if provided
    if negative_dir and os.path.exists(negative_dir):
        neg_sequences, neg_labels = load_dynamic_data(negative_dir, label=0,
                                                     window_size=window_size,
                                                     step_size=step_size)
        all_sequences.extend(neg_sequences)
        all_labels.extend(neg_labels)
        print(f"Loaded {len(neg_sequences)} negative windows")
    
    if len(all_sequences) == 0:
        raise ValueError("No valid sequences found in the data directories")
    
    # Get original sequence lengths before padding
    original_lengths = [len(seq) for seq in all_sequences]
    
    # Split sequences into train/test BEFORE normalization to prevent data leakage
    indices = list(range(len(all_sequences)))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - test_split))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_sequences = [all_sequences[i] for i in train_indices]
    test_sequences = [all_sequences[i] for i in test_indices]
    train_labels_list = [all_labels[i] for i in train_indices]
    test_labels_list = [all_labels[i] for i in test_indices]
    train_lengths_list = [original_lengths[i] for i in train_indices]
    test_lengths_list = [original_lengths[i] for i in test_indices]
    
    # Compute feature statistics from TRAINING data only (for model embedding)
    print("Computing feature normalization statistics from training data...")
    mean, std = compute_feature_statistics(train_sequences)
    print(f"Feature means: {mean}")
    print(f"Feature stds: {std}")
    
    # Pad sequences to same length (use window_size as max length for windowed data)
    train_padded = pad_sequences(train_sequences, window_size)  # Raw data
    test_padded = pad_sequences(test_sequences, window_size)    # Raw data
    
    # Convert to tensors
    train_data = torch.tensor(train_padded, dtype=torch.float32)  # Shape: (num_samples, seq_length, features)
    train_labels = torch.tensor(train_labels_list, dtype=torch.float32)        # Shape: (num_samples,) - one label per sequence
    train_lengths = torch.tensor(train_lengths_list, dtype=torch.long)
    
    test_data = torch.tensor(test_padded, dtype=torch.float32)
    test_labels = torch.tensor(test_labels_list, dtype=torch.float32)
    test_lengths = torch.tensor(test_lengths_list, dtype=torch.long)
    
    print(f"\nTotal windows created: {len(all_sequences)}")
    print(f"Training windows: {len(train_data)}, Test windows: {len(test_data)}")
    print("Normalization statistics computed (will be embedded in model)")
    print("Data saved as RAW values (not normalized)")
    
    # Verify class balance
    train_positives = (train_labels == 1).sum().item()
    train_negatives = (train_labels == 0).sum().item()
    test_positives = (test_labels == 1).sum().item()
    test_negatives = (test_labels == 0).sum().item()
    
    print(f"\nClass balance:")
    print(f"Training set: {train_positives} positives, {train_negatives} negatives")
    print(f"Test set: {test_positives} positives, {test_negatives} negatives")
    
    if train_negatives == 0 or test_negatives == 0:
        print("WARNING: No negative examples found! Model will not learn properly.")
        print("Check that negative data files exist and have proper format.")
    
    return train_data, train_labels, train_lengths, test_data, test_labels, test_lengths, mean, std

def main():
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Process dynamic gesture data
    positive_dir = os.path.join(SCRIPT_DIR, "data(Dynamic)/pos")
    negative_dir = os.path.join(SCRIPT_DIR, "data(Dynamic)/neg")
    
    print("Processing dynamic gesture data...")
    train_data, train_labels, train_lengths, test_data, test_labels, test_lengths, mean, std = create_sequence_tensors(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        test_split=0.2,
        overlap_ratio=0.5
    )
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Features per timestep: {train_data.shape[2]}")
    
    # Save tensors
    train_data_dir = os.path.join(SCRIPT_DIR, "train_data")
    test_data_dir = os.path.join(SCRIPT_DIR, "test_data")
    
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    
    torch.save({'data': train_data, 'labels': train_labels, 'lengths': train_lengths}, os.path.join(train_data_dir, "train_sequences_0.pt"))
    torch.save({'data': test_data, 'labels': test_labels, 'lengths': test_lengths}, os.path.join(test_data_dir, "test_sequences_0.pt"))
    
    # Save normalization statistics
    normalization_stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    model_path = os.path.join(os.path.dirname(SCRIPT_DIR), "trained_model")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "normalization_stats.json"), "w") as f:
        json.dump(normalization_stats, f, indent=2)
    print(f"\nSaved normalization statistics to: {os.path.join(model_path, 'normalization_stats.json')}")
    
    print("\nData processing complete!")
    print(f"Saved training data to: {os.path.join(train_data_dir, 'train_sequences_0.pt')}")
    print(f"Saved testing data to: {os.path.join(test_data_dir, 'test_sequences_0.pt')}")
    print("✅ Raw data saved - normalization will be embedded in model")

if __name__ == "__main__":
    main() 