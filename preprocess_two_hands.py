import json
import os


class Bound:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def normalize(self, val):
        return (val - self.min) / (self.max - self.min)


AngleBound = Bound(min=0, max=360)
DistanceBound = Bound(min=-1, max=1)


def preprocess_file(input_file, output_dir):
    with open(input_file) as f:
        data = json.load(f)

    for position in data:
        hand_data = position["handData"]
        for i, value in enumerate(hand_data):
            if i in [5, 9, 13, 16, 22, 26, 30, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]:
                hand_data[i] = DistanceBound.normalize(value)
            else:
                hand_data[i] = AngleBound.normalize(value)
    output_file = os.path.join(output_dir, "normalized_" + os.path.basename(input_file))
    with open(output_file, "w") as f:
        json.dump(data, f)

def preprocess_directory(data_dir):
    output_dir = os.path.join(data_dir, "normalized")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            input_file = os.path.join(data_dir, filename)
            preprocess_file(input_file, output_dir)

def main():
    data_dir = "model_archive/heart_two_hands_v0.0.1/data"
    preprocess_directory(data_dir)

if __name__ == "__main__":
    main()