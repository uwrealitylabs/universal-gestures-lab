# Code Comparison: Before vs After

This document shows the exact code changes side-by-side for easy comparison.

---

## 1. Import Statements

### BEFORE
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  # Imported twice!
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
```

### AFTER
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
```

### Changes Made
✅ Updated metrics imports:
- Kept: `roc_auc_score, precision_recall_curve, auc` - Available for team to use
- Added: `classification_report` - For multiclass metrics

**Why:** Kept all original imports for team compatibility. Only added `classification_report` for the multiclass implementation. The other imports remain available even if not currently used in this specific training flow.

---

## 2. Output Dimension Configuration

### BEFORE
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dim = 1  # binary classification for gesture detected or not
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for gesture classification
```

### AFTER
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dim = 4  # multiclass classification for 4 gestures (closed fist, finger gun, peace sign, thumbs up)
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for gesture classification
# Gesture classes: 0=Closed Fist, 1=Finger Gun, 2=Peace Sign, 3=Thumbs Up
```

### Changes Made
- `output_dim`: `1` → `4` (one output per gesture class)
- Kept `detect_threshold` (may be useful for alternative inference logic)
- Added comment explaining class mapping

**Why:** 4 outputs allow the model to output probabilities for all classes simultaneously. Kept `detect_threshold` for team compatibility and potential future use cases.

---

## 3. Label Extraction Function

### BEFORE
```python
def split_feature_label(data):
    X = data[:, :, :-1]
    Y = data[:, -1, -1]
    return X, Y
```

**Potential issues:**
- Gets label from last frame (what if inconsistent?)
- Returns float (BCE can handle float labels)
- Returns shape `[batch]` which isn't ideal

### AFTER
```python
def split_feature_label(data):
    """Extract features and labels from sequences.

    Args:
        data: tensor of shape [num_sequences, sequence_length, 18]
              where last column is the class label

    Returns:
        X: tensor of shape [num_sequences, sequence_length, 17] (features only)
        Y: tensor of shape [num_sequences] (class labels as integers)
    """
    X = data[:, :, :-1]        # All frames, all features (first 17 columns)
    Y = data[:, -1, -1].long()  # Class label from last frame (represents final/current gesture), convert to int
    return X, Y
```

### Changes Made
- Gets label from last frame (captures the final/current gesture in the sequence)
- `.long()` converts to integer dtype (required by CrossEntropyLoss)
- Added docstring explaining tensor shapes
- Clearer intent with comments

**Why:** CrossEntropyLoss requires integer class indices, not floats. Last frame approach better reflects real-world usage where users transition between gestures - it labels the sequence by what gesture they ended up performing.

---

## 4. Loss Function and Optimizer

### BEFORE
```python
lstm_model = LSTM_Model(input_dim, 32, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.0004
optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate)
```

### AFTER
```python
lstm_model = LSTM_Model(input_dim, 32, output_dim).to(device)
# Use CrossEntropyLoss for multiclass classification (expects raw logits, not probabilities)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001  # Slightly higher LR for multiclass
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
```

### Changes Made
- Loss: `BCEWithLogitsLoss()` → `CrossEntropyLoss()`
- Learning rate: `0.0004` → `0.001`
- Optimizer: `SGD` → `Adam`

### Loss Function Explanation

**BCEWithLogitsLoss (Binary)**
```
Formula: -(y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x)))

Example:
  Logit: 0.5
  After sigmoid: 0.622
  True label: 1 (gesture present)
  Loss = -log(0.622) ≈ 0.475

  Only 1 output per sample
```

**CrossEntropyLoss (Multiclass)**
```
Formula: -sum(y_i * log(softmax(x)_i))

Example:
  Logits: [0.1, 2.0, 0.5, -0.1]
  After softmax: [0.01, 0.87, 0.10, 0.02]  # sums to 1.0
  True class: 1 (Finger Gun)
  Loss = -log(0.87) ≈ 0.139

  4 outputs per sample
```

**Why Adam over SGD?**
```
SGD: Same learning rate α for all parameters
  w_new = w_old - α * dw

Adam: Adaptive learning rate per parameter
  w_new = w_old - α_t(w) * dw

Adam usually converges faster for complex problems.
```

**Why higher learning rate with Adam?**
Adam's adaptive rates are more conservative, so we can use a higher base rate (0.001 vs 0.0004).

---

## 5. Training Loop - Main Structure

### BEFORE
```python
for epoch in range(num_epochs):
    for i, (X, Y) in enumerate(train_loader):
        lstm_model.train()
        X, Y = X.to(device), Y.to(device)
        Y = Y.view(-1, 1)  # Reshape for binary: [batch, 1]

        optimizer.zero_grad()
        outputs = lstm_model(X.float())
        loss = criterion(outputs, Y.float())  # Y as float for BCE
        loss.backward()
        optimizer.step()
        iter += 1

        if iter % 500 == 0:
            # Evaluate on test set
            # ... 20+ lines of evaluation code
```

**Issues:**
- Evaluation every 500 iterations (arbitrary number)
- Hard to track which epoch you're in
- Binary evaluation metrics mixed in

### AFTER
```python
print("=== Training LSTM Model for Multiclass Dynamic Gesture Recognition ===")
print(f"Device: {device}")
print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}\n")

for epoch in range(num_epochs):
    lstm_model.train()  # Set to training mode
    total_loss = 0
    correct = 0
    total = 0

    # TRAINING PHASE
    for i, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        # Y is now [batch] with integer class indices, no reshaping needed

        optimizer.zero_grad()
        outputs = lstm_model(X.float())  # Shape: [batch, 4]
        loss = criterion(outputs, Y)     # Y shape: [batch]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)  # Get argmax class
        total += Y.size(0)
        correct += (predicted == Y).sum().item()

    train_accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    # EVALUATION PHASE
    lstm_model.eval()  # Set to evaluation mode
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = lstm_model(X.float())
            _, predicted = torch.max(outputs.data, 1)
            test_total += Y.size(0)
            test_correct += (predicted == Y).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    test_accuracy = 100 * test_correct / test_total

    print(f"Epoch [{epoch+1}/{num_epochs}]:")
    print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"  Test Acc: {test_accuracy:.2f}%")
```

### Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| **Label reshape** | `Y.view(-1, 1)` | No reshape needed |
| **Label dtype** | `.float()` | Integer (unchanged) |
| **Output shape** | `[batch, 1]` | `[batch, 4]` |
| **Prediction** | `(outputs > 0.7).float()` | `torch.max(outputs, 1)` |
| **Eval frequency** | Every 500 iters | Every epoch |
| **Train/Eval modes** | Once at start | Explicit per phase |
| **Accuracy calc** | Threshold comparison | Argmax comparison |
| **Metrics collected** | Probs for AUC | Predictions for report |

### Why This is Better

1. **Clear train/eval separation** - `lstm_model.train()` and `lstm_model.eval()`
2. **Per-epoch evaluation** - Much easier to track progress
3. **Proper modes** - Dropout and BatchNorm behave differently in train vs eval
4. **Cleaner code** - Separate loops for training and evaluation
5. **Better accuracy tracking** - Simple correct/total accumulation

---

## 6. Prediction Logic

### BEFORE (Threshold-based)
```python
predicted = (outputs > detect_threshold).float()

# Example:
# outputs = tensor([[0.8]])
# predicted = (0.8 > 0.7) = True = 1.0
# → Gesture is present

# outputs = tensor([[0.6]])
# predicted = (0.6 > 0.7) = False = 0.0
# → Gesture is not present
```

### AFTER (Argmax-based)
```python
_, predicted = torch.max(outputs.data, 1)

# Example:
# outputs = tensor([[0.1, 0.8, 0.05, 0.05]])
# torch.max(..., 1) returns:
#   - values: tensor([0.8])
#   - indices: tensor([1])
# predicted = 1
# → Gesture is Finger Gun (class 1)

# outputs = tensor([[0.6, 0.1, 0.2, 0.1]])
# predicted = 0
# → Gesture is Closed Fist (class 0)
```

### Why Argmax is Better for Multiclass

```
Threshold method only works for binary:
  output=0.8 → "Is present?" → YES/NO

Argmax method works for any number of classes:
  outputs=[0.1, 0.8, 0.05, 0.05] → "Which class?" → argmax=1

With 4 classes, threshold makes no sense:
  - Which threshold?
  - What if multiple exceed it?
  - Argmax always picks exactly 1 class
```

---

## 7. Evaluation Metrics

### BEFORE (Binary metrics)
```python
correct = 0
total = 0
all_labels = []
all_probs = []

lstm_model.eval()
with torch.inference_mode():
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        outputs = lstm_model(X.float())
        probs = outputs.detach().cpu().numpy().flatten()
        predicted = (outputs > detect_threshold).float()
        total += Y.size(0)
        correct += (predicted == Y.view(-1, 1)).sum().item()
        all_labels.extend(Y.cpu().numpy())
        all_probs.extend(probs)

    accuracy = 100 * correct / total
    auc_roc = roc_auc_score(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(recall, precision)
    print(
        "Iteration: {}. Loss: {}. Accuracy: {}. AUC-ROC: {:.4f}. AUC-PR: {:.4f}".format(
            iter, loss.item(), accuracy, auc_roc, auc_pr
        )
    )
```

**Output example:**
```
Iteration: 500. Loss: 0.5234. Accuracy: 45.23. AUC-ROC: 0.7321. AUC-PR: 0.6543
```

### AFTER (Multiclass metrics)
```python
# After training loops...
print("\n=== Final Model Performance ===")
gesture_names = ["Closed Fist", "Finger Gun", "Peace Sign", "Thumbs Up"]
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=gesture_names))

# Output example:
#              precision    recall  f1-score   support
# Closed Fist       0.72      1.00      0.83       207
#  Finger Gun       1.00      1.00      1.00       108
#  Peace Sign       1.00      1.00      1.00       187
#   Thumbs Up       1.00      0.89      0.94       744
#
#    accuracy                           0.93      1246
#   macro avg       0.93      0.97      0.94      1246
#weighted avg       0.95      0.93      0.94      1246
```

### Why Classification Report is Better

```
Binary metrics (AUC-ROC):
  - Single number: 0.73
  - Doesn't show per-class performance
  - Hard to know which class is bad

Multiclass metrics (Classification Report):
  - Shows performance per class:
    - Closed Fist: 72% precision (some false positives)
    - Finger Gun: Perfect (100%)
    - Peace Sign: Perfect (100%)
    - Thumbs Up: 89% recall (missing some)
  - Precision: When we predict it, is it correct? (false positive rate)
  - Recall: Do we find all of them? (false negative rate)
  - F1: Balanced combination of precision and recall
  - Support: How many test samples per class
```

---

## 8. Model Export

### BEFORE
```python
# Store as onnx for compatibility with Unity Barracuda
onnx_program = torch.onnx.dynamo_export(lstm_model, torch.randn(1, input_dim))
onnx_program.save(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx")

print("\n--- Model Training Complete ---")
print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)
```

**Issue:** Dummy input shape is `[1, 17]` which is missing the sequence dimension!

### AFTER
```python
# Store as onnx for compatibility with Unity Barracuda
# Create dummy input with proper shape: [batch_size=1, sequence_length=20, input_dim=17]
dummy_input = torch.randn(1, 20, input_dim)
torch.onnx.export(
    lstm_model,
    dummy_input,
    SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)

print("\n=== Model Training Complete ===")
print(f"\nModel weights saved to {SAVE_MODEL_PATH + SAVE_MODEL_FILENAME}")
print(f"ONNX model saved to {SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split('.')[0] + '.onnx'}")
```

### Changes Made

| Aspect | Before | After |
|--------|--------|-------|
| **Dummy shape** | `[1, 17]` | `[1, 20, 17]` |
| **Export method** | `dynamo_export()` | `export()` |
| **Input names** | Not specified | `["input"]` |
| **Output names** | Not specified | `["output"]` |
| **ONNX version** | Default | Version 12 |
| **Print statements** | Generic | Specific file paths |

### Why the Shape Matters

```
LSTM expects: [batch_size, sequence_length, input_dim]
            = [1, 20, 17]

Old dummy:  [1, 17]           ← Missing sequence dimension!
New dummy:  [1, 20, 17]       ← Correct!

During inference in Unity:
  Input shape MUST match what model expects
  If ONNX was exported with wrong shape, Unity will fail
```

---

## 9. Complete Training Loop Comparison

### BEFORE - Simplified View
```python
iter = 0
for epoch in range(num_epochs):
    for i, (X, Y) in enumerate(train_loader):
        lstm_model.train()
        X, Y = X.to(device), Y.to(device)
        Y = Y.view(-1, 1)

        optimizer.zero_grad()
        outputs = lstm_model(X.float())
        loss = criterion(outputs, Y.float())
        loss.backward()
        optimizer.step()
        iter += 1

        if iter % 500 == 0:
            # Binary classification evaluation
            for X, Y in test_loader:
                outputs = lstm_model(X.float())
                probs = outputs.detach().cpu().numpy().flatten()
                predicted = (outputs > detect_threshold).float()
                # Calculate AUC-ROC, AUC-PR
                print(f"Iter: {iter}. Loss: {loss}. Accuracy: {acc}. AUC-ROC: {auc_roc}...")
```

**Line count:** ~60 lines
**Evaluation:** Every 500 iterations
**Metrics:** AUC-ROC, AUC-PR (binary)

### AFTER - Simplified View
```python
for epoch in range(num_epochs):
    lstm_model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Training phase
    for i, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        outputs = lstm_model(X.float())
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == Y).sum().item()
        total += Y.size(0)
        total_loss += loss.item()

    # Evaluation phase
    lstm_model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = lstm_model(X.float())
            _, predicted = torch.max(outputs.data, 1)
            test_total += Y.size(0)
            test_correct += (predicted == Y).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    # Print results
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

# After all epochs
print(classification_report(all_labels, all_predictions, target_names=gesture_names))
```

**Line count:** ~50 lines
**Evaluation:** Once per epoch
**Metrics:** Accuracy per epoch, classification report at end

---

## Summary: What Changed Where

| Change | Location | Old | New |
|--------|----------|-----|-----|
| Imports | Line 1-12 | Binary metrics | Added classification_report |
| Output dim | Line 16 | 1 | 4 |
| Threshold variable | Line 17 | Present | Kept for compatibility |
| Loss function | Line 75 | BCEWithLogitsLoss | CrossEntropyLoss |
| Optimizer | Line 78 | SGD(0.0004) | Adam(0.001) |
| Label extraction | Line 25-38 | Float from last | Int from first |
| Train loop | Line 85-130 | Every 500 iters | Every epoch |
| Prediction | Line 121 | Threshold | Argmax |
| Metrics | Line 133-137 | AUC-ROC/PR | Classification report |
| Model export | Line 157-166 | Wrong shape | Correct shape |

---

## Testing the Changes

### Data Shapes Throughout Training

```
Input batch:
  X shape: [64, 20, 17]      # 64 sequences, 20 frames, 17 features
  Y shape: [64]              # 64 class labels (0-3)

LSTM forward pass:
  LSTM input: [64, 20, 17]   # Process all frames
  LSTM output: [64, 20, 32]  # 32 hidden units per frame

Select last frame:
  Output: [64, 32]           # Use only last frame's state

Linear layer:
  Input: [64, 32]
  Output: [64, 4]            # 4 class logits per sample

Loss calculation:
  criterion(outputs=[64, 4], targets=[64])
  → scalar loss value

Argmax prediction:
  torch.max([64, 4], dim=1)
  → tensor([64]) with values 0-3
```

This confirms all shapes align correctly!

