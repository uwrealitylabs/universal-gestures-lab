import json
import os
import random


# -------------------------------------------------- #
# Define the target answers for each class


LABELS = 4
TRAIN_SPLIT = 0.6  # 60% for training, 40% for testing

target_answers = {
    0: ["closedFistPositive.json"],
    1: ["fingerGunPositive.json"],
    2: ["peaceSignPositive.json"],
    3: ["thumbsupJustin.json", "thumbsuplily.json", "thumbsupNathan.json"],
}

# -------------------------------------------------- #


PARENT_FOLDER = "src/data"
TARGET_TRAIN_FOLDER = "src/train_data"
TARGET_TEST_FOLDER = "src/test_data"

# Create target folders if they don't exist
os.makedirs(TARGET_TRAIN_FOLDER, exist_ok=True)
os.makedirs(TARGET_TEST_FOLDER, exist_ok=True)

for i, files in target_answers.items():
    all_class_data = []

    # First, collect all data for this class
    for fname in files:
        source_path = os.path.join(PARENT_FOLDER, fname)

        # check if source exists
        if not os.path.exists(source_path):
            print(f"Source file {source_path} does not exist.")
            continue

        # load source data
        with open(source_path, "r") as f:
            data = json.load(f)

            # Modify each item in the data array
            for item in data:
                if isinstance(item, dict) and "confidence" in item:
                    # Replace scalar confidence with class index for multi-class
                    item["label"] = i  # Set the correct class index
                    del item["confidence"]  # Remove the scalar confidence
                    all_class_data.append(item)

    if not all_class_data:
        print(f"No data found for class {i}")
        continue

    # Shuffle the data for this class
    random.shuffle(all_class_data)

    # Split into train and test
    train_size = int(len(all_class_data) * TRAIN_SPLIT)
    train_data = all_class_data[:train_size]
    test_data = all_class_data[train_size:]

    # Save training data
    train_path = os.path.join(TARGET_TRAIN_FOLDER, f"train_{i}.json")
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    # Save test data
    test_path = os.path.join(TARGET_TEST_FOLDER, f"test_{i}.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(
        f"Class {i}: {len(train_data)} training samples, {len(test_data)} test samples"
    )
    print(f"  Train data saved to: {train_path}")
    print(f"  Test data saved to: {test_path}")

print("Data splitting complete!")
