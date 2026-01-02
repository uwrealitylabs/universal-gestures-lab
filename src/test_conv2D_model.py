import torch
import json
import os
import numpy as np
import argparse
import glob
import random
from model_dynamic import Conv2D_Model

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_model", "model_dynamic_weights.json")
NORM_STATS_PATH = os.path.join(PROJECT_ROOT, "trained_model", "normalization_stats.json")
DATA_DIR = os.path.join(SCRIPT_DIR, "data(Dynamic)")

def load_model():
    """Load the trained Conv2D Unity model"""
    # Load normalization stats
    with open(NORM_STATS_PATH, 'r') as f:
        norm_stats = json.load(f)
    
    # Create model
    model = Conv2D_Model(
        input_dim=17,
        output_dim=1,
        norm_mean=norm_stats['mean'],
        norm_std=norm_stats['std']
    )
    
    # Load weights
    with open(MODEL_PATH, 'r') as f:
        state_dict_json = json.load(f)
    
    # Convert lists back to tensors
    state_dict = {}
    for key, value in state_dict_json.items():
        state_dict[key] = torch.tensor(value)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def test_with_json_file(model, json_path):
    """Test model with a JSON file from Unity"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both single sequence and list of sequences
    if isinstance(data, list):
        # Take first sequence if it's a list
        data = data[0]
    
    # Extract confidence (label)
    confidence = data['confidence']
    
    # Process sequence data - it's stored as a flat list in sequenceData
    sequence_data = data['sequenceData']
    
    # Convert to tensor (15 timesteps x 17 features)
    sequence_tensor = torch.tensor(sequence_data[:15], dtype=torch.float32)
    
    # Add batch and width dimensions for 4D NHWC format
    # (15, 17) -> (1, 15, 1, 17)
    input_tensor = sequence_tensor.unsqueeze(0).unsqueeze(2)
    
    print(f"Input shape: {input_tensor.shape} (batch=1, height=15, width=1, channels=17)")
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    
    predicted = "POSITIVE" if prob > 0.7 else "NEGATIVE"
    expected = "POSITIVE" if confidence == 1 else "NEGATIVE"
    
    print(f"Model output (logit): {output.item():.4f}")
    print(f"Probability: {prob:.4f}")
    print(f"Prediction: {predicted}")
    print(f"Expected: {expected}")
    print(f"Result: {'[CORRECT]' if predicted == expected else '[INCORRECT]'}")
    
    return predicted == expected

def get_available_files(directory):
    """Get all JSON files from a directory"""
    pattern = os.path.join(directory, "*.json")
    return glob.glob(pattern)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Conv2D Unity Model with dynamic gesture data')
    parser.add_argument('--all', action='store_true', 
                       help='Test all available files instead of just one example')
    parser.add_argument('--random', action='store_true',
                       help='Select random files for testing')
    parser.add_argument('--gesture', type=str, default=None,
                       help='Filter files by gesture name (e.g., "fire_finger_gun")')
    parser.add_argument('--file', type=str, default=None,
                       help='Test a specific file')
    args = parser.parse_args()
    
    print("Testing Conv2D Unity Model")
    print("=" * 50)
    
    # Load model
    model = load_model()
    print("[OK] Model loaded successfully\n")
    
    # Test specific file if provided
    if args.file:
        if os.path.exists(args.file):
            print(f"Testing specific file: {os.path.basename(args.file)}")
            print("-" * 30)
            test_with_json_file(model, args.file)
        else:
            print(f"[ERROR] File not found: {args.file}")
        return
    
    # Get available files
    pos_dir = os.path.join(DATA_DIR, "pos")
    neg_dir = os.path.join(DATA_DIR, "neg")
    
    pos_files = get_available_files(pos_dir)
    neg_files = get_available_files(neg_dir)
    
    # Filter by gesture name if specified
    if args.gesture:
        pos_files = [f for f in pos_files if args.gesture in os.path.basename(f)]
        neg_files = [f for f in neg_files if args.gesture in os.path.basename(f)]
    
    if not pos_files and not neg_files:
        print("[ERROR] No test files found!")
        if args.gesture:
            print(f"   No files matching gesture: {args.gesture}")
        print(f"   Checked directories:")
        print(f"   - {pos_dir}")
        print(f"   - {neg_dir}")
        return
    
    # Determine which files to test
    if args.all:
        # Test all files
        files_to_test = [(f, "POSITIVE") for f in pos_files] + [(f, "NEGATIVE") for f in neg_files]
        print(f"Testing ALL files ({len(files_to_test)} total)\n")
    else:
        files_to_test = []
        
        # Select positive examples
        if pos_files:
            if args.random:
                selected_pos = [random.choice(pos_files)]
            else:
                selected_pos = [pos_files[0]]  # First file
            files_to_test.extend([(f, "POSITIVE") for f in selected_pos])
        
        # Select negative examples
        if neg_files:
            if args.random:
                selected_neg = [random.choice(neg_files)]
            else:
                selected_neg = [neg_files[0]]  # First file
            files_to_test.extend([(f, "NEGATIVE") for f in selected_neg])
        
        print(f"Testing sample files ({'random' if args.random else 'first available'})\n")
    
    # Test files and track results
    results = []
    for file_path, expected_label in files_to_test:
        filename = os.path.basename(file_path)
        print(f"Testing {expected_label} example: {filename}")
        print("-" * 30)
        
        correct = test_with_json_file(model, file_path)
        results.append((filename, correct))
        print()
    
    # Summary
    print("=" * 50)
    if len(results) > 2:  # Only show summary for multiple files
        correct_count = sum(1 for _, correct in results if correct)
        total_count = len(results)
        accuracy = correct_count / total_count * 100
        print(f"Summary: {correct_count}/{total_count} correct ({accuracy:.1f}% accuracy)")
        
        # Show incorrect files if any
        incorrect = [name for name, correct in results if not correct]
        if incorrect:
            print("\n[ERROR] Incorrect predictions:")
            for name in incorrect:
                print(f"   - {name}")
    
    print("[OK] Testing complete. Model maintains NHWC format throughout.")

if __name__ == "__main__":
    main()