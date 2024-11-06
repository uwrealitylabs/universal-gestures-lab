import json
import numpy as np
import os

# Parameters for valid range and distance margin
CURL_RANGE = (180, 260)
DISTANCE_MARGIN = 10

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def calculate_avg_positive_values(positive_filepath):

    positive_data = load_json(positive_filepath)
    avg_values = {}
    num_samples = len(positive_data)
    
    for sample in positive_data:
        for i, value in enumerate(sample["handData"]):
            avg_values[i] = avg_values.get(i, 0) + value
    
    for finger in avg_values:
        avg_values[finger] /= num_samples
    
    return avg_values

def generate_random_data(avg_positive_values, num_samples=10):

    synthetic_data = []
    for _ in range(num_samples):
        example = []
        for finger, avg_value in avg_positive_values.items():
            rand_value = np.random.uniform(CURL_RANGE[0], CURL_RANGE[1])
            while abs(rand_value - avg_value) < DISTANCE_MARGIN:
                rand_value = np.random.uniform(CURL_RANGE[0], CURL_RANGE[1])
            example.append(rand_value)
        thing = {"confidence": 0, "handData": example, }
        synthetic_data.append(thing)
        print(synthetic_data[0])
    return synthetic_data

def augment_negative_data(negative_filepath, positive_filepath, output_filepath, num_synthetic_samples=10):

    negative_data = load_json(negative_filepath)
    avg_positive_values = calculate_avg_positive_values(positive_filepath)
    
    # Generate synthetic data and append to negative examples
    synthetic_data = generate_random_data(avg_positive_values, num_synthetic_samples)
    negative_data.extend(synthetic_data)

    # Save the augmented data to the place u want it
    save_json(negative_data, output_filepath)
    print(f"Augmented negative data saved to {output_filepath}")

def main():
    # file paths
    negative_filepath = "data/thumbsupNegJustin.json"
    positive_filepath = "data/thumbsupJustin.json"  # Raw positive gesture data 
    output_filepath = "data/thumbsupNegDataJustinAugmented.json"
    num_synthetic_samples = 10  # nnum of synthetic samples to generate per run
    
    # Run augmentation process
    augment_negative_data(negative_filepath, positive_filepath, output_filepath, num_synthetic_samples)

if __name__ == "__main__":
    main()
