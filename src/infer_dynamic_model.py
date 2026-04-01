from pathlib import Path
import torch

from src.dynamic_model_utils import (
    TARGET_FRAME_SIZE,
    MODEL_WEIGHTS_PATH,
    load_dynamic_sample,
    load_class_names,
    DynamicGestureClassifier,
)


def predict_file(file_path: str):
    class_names = load_class_names()

    model = DynamicGestureClassifier(
        input_dim=TARGET_FRAME_SIZE,
        num_classes=len(class_names),
    )
    model.load_state_dict(
    torch.load(MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    sample = load_dynamic_sample(Path(file_path))
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        topk = min(4, len(class_names))
        top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)

    print("Probabilities:", probs)
    print("Top predictions:")
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        print(f"  {class_names[idx.item()]}: {prob.item():.4f}")

    predicted_class = class_names[top_idxs[0][0].item()]
    print("Predicted class:", predicted_class)


if __name__ == "__main__":
    test_file = "src/data(Dynamic)/pos/dynamic_fire_finger_gun_pos1_2025-08-22_16-41-08.json"
    predict_file(test_file)