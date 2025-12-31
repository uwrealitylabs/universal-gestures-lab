import torch
import os
import json
from model_dynamic_multiclass import Conv2D_MultiClass_Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_model")
SAVE_MODEL_FILENAME = "model_dynamic_multiclass_weights.json"
NORM_STATS_PATH = os.path.join(SAVE_MODEL_PATH, "normalization_stats_multiclass.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3
input_dim = 17

# Load normalization stats
with open(NORM_STATS_PATH, 'r') as f:
    norm_stats = json.load(f)
    norm_mean = norm_stats['mean']
    norm_std = norm_stats['std']

# Create model
model = Conv2D_MultiClass_Model(input_dim, num_classes, norm_mean=norm_mean, norm_std=norm_std).to(device)

# Load trained weights
with open(os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME), 'r') as f:
    state_dict_json = json.load(f)
    state_dict = {k: torch.tensor(v) for k, v in state_dict_json.items()}
    model.load_state_dict(state_dict)

model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 15, 1, input_dim).to(device)
onnx_filename = SAVE_MODEL_FILENAME.replace('.json', '.onnx')

torch.onnx.export(
    model,
    dummy_input,
    os.path.join(SAVE_MODEL_PATH, onnx_filename),
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    export_params=True,
    opset_version=12,
    do_constant_folding=True
)

print(f"ONNX model exported to: {SAVE_MODEL_PATH}/{onnx_filename}")
