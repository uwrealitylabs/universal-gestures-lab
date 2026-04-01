import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.dynamic_model_utils import (
    TARGET_FRAME_SIZE,
    MODEL_WEIGHTS_PATH,
    load_dataset,
    save_class_names,
    DynamicGestureClassifier,
)

BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-3


def main():
    X, y, class_names = load_dataset()

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Class names:", class_names)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DynamicGestureClassifier(
        input_dim=TARGET_FRAME_SIZE,
        num_classes=len(class_names),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")

    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    save_class_names(class_names)
    print(f"Saved weights to: {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()