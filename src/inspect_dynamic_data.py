import json
from pathlib import Path

# Change this to an actual file inside src/data(Dynamic)/...
file_path = Path("src/data(Dynamic)/pos")

# find first json file
json_files = list(file_path.glob("*.json"))
if not json_files:
    raise FileNotFoundError(f"No JSON files found in {file_path}")

sample_file = json_files[0]
print("Using file:", sample_file)

with open(sample_file, "r") as f:
    data = json.load(f)

print("Top-level type:", type(data))
print("Number of frames:", len(data))

first_frame = data[0]
print("Keys in first frame:", first_frame.keys())
print("Confidence value:", first_frame["confidence"])
print("Length of sequenceData:", len(first_frame["sequenceData"]))
print("First 10 sequenceData values:", first_frame["sequenceData"][:10])