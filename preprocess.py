import json
import argparse
import os


class Bound:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def normalize(self, val):
        return (val - self.min) / (self.max - self.min)


CurlBound = Bound(min=180, max=260)
FlexionBound = Bound(min=180, max=260)
AbductionBound = Bound(min=8, max=90)
OppositionBound = Bound(min=0, max=0.2)


def is_json(parser, file):
    if not file.endswith(".json"):
        parser.error("File must be a json")
    else:
        return file


def is_dir(parser, path):
    if not os.path.exists(path):
        parser.error(path + " is not a valid location")
    else:
        return path


parser = argparse.ArgumentParser(description="Normalize hand measurements")
parser.add_argument(
    "input", type=lambda f: is_json(parser, f), help="Json file containing measurements"
)
parser.add_argument(
    "file_path",
    type=lambda f: is_dir(parser, f),
    help="Folder path where normalized Json will be written",
)

args = parser.parse_args()
input = args.input
with open(input) as f:
    data = json.load(f)

for position in data:
    hand_data = position["handData"]
    for i, value in enumerate(position["handData"]):
        value = float(value)
        match i:
            case 0 | 4 | 6 | 10 | 14:
                hand_data[i] = CurlBound.normalize(value)
            case 1 | 5 | 7 | 11:
                hand_data[i] = AbductionBound.normalize(value)
            case 2 | 8 | 12 | 15:
                hand_data[i] = FlexionBound.normalize(value)
            case 3 | 9 | 13 | 16:
                hand_data[i] = OppositionBound.normalize(value)

with open(args.file_path + "normalized_" + input, "w") as f:
    json.dump(data, f)
