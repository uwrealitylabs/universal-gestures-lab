# Quick Reference - What Was Changed

## 📋 Document Guide

The full explanation is in: **`MULTICLASS_DYNAMIC_IMPLEMENTATION.md`** (898 lines)

**Best way to read it:**
1. Start with "Problem Overview" - understand what was wrong
2. Read "Understanding the Existing Systems" - see how static model works
3. Go through "Files Created" and "Files Modified" - see what changed
4. Study "Detailed Code Changes" - understand the edits line-by-line
5. Check "Training and Results" - see it actually works

---

## 🔄 The Big Picture: What Changed

### Before (Binary Classification)
```
1 Gesture Recognition Model
├─ Closed Fist? YES/NO
├─ Finger Gun? YES/NO
├─ Peace Sign? YES/NO
└─ Thumbs Up? YES/NO
= Need 4 separate ONNX files!
```

### After (Multiclass Classification)
```
1 Gesture Recognition Model
├─ Is it Closed Fist?  [probability]
├─ Is it Finger Gun?   [probability]
├─ Is it Peace Sign?   [probability]
└─ Is it Thumbs Up?    [probability]
= Pick the highest probability!
= Only 1 ONNX file!
```

---

## 📁 Files Changed

### Created: `process_dynamic_data.py`
**Lines of code:** ~140
**What it does:**
- Loads JSON gesture data
- Creates 20-frame sliding windows
- Adds class labels (0-3)
- Saves as PyTorch tensors

**Key line:** `SEQUENCE_LENGTH = 20`

### Modified: `model_dynamic.py`
**Total changes:** ~60 lines across 9 different edits

| Change | Old | New |
|--------|-----|-----|
| **Output Dim** | `output_dim = 1` | `output_dim = 4` |
| **Loss Function** | `BCEWithLogitsLoss()` | `CrossEntropyLoss()` |
| **Optimizer** | `SGD (0.0004)` | `Adam (0.001)` |
| **Prediction** | `outputs > 0.7` | `torch.max(outputs, 1)` |
| **Metrics** | AUC-ROC, AUC-PR | Classification Report |
| **Label Shape** | `[batch, 1]` float | `[batch]` int |

---

## 🔑 Key Code Changes Explained

### Change 1: Output Dimension
```python
# OLD - Binary (detect or not detect)
output_dim = 1

# NEW - Multiclass (which class?)
output_dim = 4  # One for each gesture
```
**Why:** 4 outputs = 4 classes (one probability per gesture)

---

### Change 2: Loss Function
```python
# OLD - Binary Cross Entropy
criterion = nn.BCEWithLogitsLoss()
# Formula: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
# Input: 1 value (0 or 1)

# NEW - Categorical Cross Entropy
criterion = nn.CrossEntropyLoss()
# Formula: -Σ[y_i * log(softmax(x)_i)]
# Input: 4 values (class 0, 1, 2, or 3)
```

**Visual example:**
```
Binary output:     [0.7]
Probability: gesture present (0.7), absent (0.3)

Multiclass output: [0.1, 0.8, 0.05, 0.05]
Probabilities:
  - Class 0: 0.1  (Closed Fist)
  - Class 1: 0.8  (Finger Gun) ← WINNER!
  - Class 2: 0.05 (Peace Sign)
  - Class 3: 0.05 (Thumbs Up)
```

---

### Change 3: Prediction Method
```python
# OLD - Threshold-based (binary)
predicted = (outputs > 0.7).float()
# Is probability > 0.7? Yes = class 1, No = class 0

# NEW - Argmax-based (multiclass)
_, predicted = torch.max(outputs, 1)
# Which class has highest probability? Pick it!
```

**Example:**
```python
# Old method
output = 0.75
predicted = 1  (present)

# New method
outputs = [0.1, 0.8, 0.05, 0.05]
_, predicted = torch.max(outputs, 1)
predicted = 1  (Finger Gun has 0.8, highest)
```

---

### Change 4: Label Handling
```python
# OLD - Float label from last frame
Y = data[:, -1, -1]  # type: float, shape: [batch]

# NEW - Integer label from first frame
Y = data[:, 0, -1].long()  # type: long (int), shape: [batch]
```

**Why:** CrossEntropyLoss requires integer class indices, not floats

---

### Change 5: Optimizer
```python
# OLD - Simple gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.0004)
# Same learning rate for all parameters
# Works but slower for complex problems

# NEW - Adaptive learning rates
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Learning rate adapts per parameter
# Better for multiclass, faster convergence
```

---

### Change 6: Training Loop Structure
```python
# OLD - Evaluate every 500 iterations
for epoch in range(num_epochs):
    for batch_idx, (X, Y) in enumerate(train_loader):
        # ... train ...
        iter += 1
        if iter % 500 == 0:
            # Evaluate and print metrics

# NEW - Evaluate every epoch
for epoch in range(num_epochs):
    # Training phase
    for batch_idx, (X, Y) in enumerate(train_loader):
        # ... train ...

    # Evaluation phase (after all batches)
    for batch_idx, (X, Y) in enumerate(test_loader):
        # ... evaluate ...
    # Print metrics once per epoch
```

**Why:** Easier to interpret, one number per epoch vs scattered every 500 iters

---

### Change 7: Evaluation Metrics
```python
# OLD - Binary classification metrics
auc_roc = roc_auc_score(labels, probs)
auc_pr = auc(recall, precision)
print(f"AUC-ROC: {auc_roc}, AUC-PR: {auc_pr}")

# NEW - Multiclass classification report
print(classification_report(labels, predictions))
# Output:
#           precision  recall  f1-score  support
# Class 0:    0.72     1.00     0.83      207
# Class 1:    1.00     1.00     1.00      108
# Class 2:    1.00     1.00     1.00      187
# Class 3:    1.00     0.89     0.94      744
```

---

### Change 8: Model Export
```python
# OLD - Wrong input shape
dummy_input = torch.randn(1, input_dim)  # [1, 17]
onnx_program = torch.onnx.dynamo_export(model, dummy_input)

# NEW - Correct input shape with sequence dimension
dummy_input = torch.randn(1, 20, input_dim)  # [1, 20, 17]
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)
```

**Why:** LSTM needs [batch, sequence_length, features], not just [batch, features]

---

## 📊 Training Results Breakdown

```
Final Test Accuracy: 93.42%

Per-Gesture Performance:
┌─────────────┬───────────┬────────┬──────┐
│ Gesture     │ Precision │ Recall │ F1   │
├─────────────┼───────────┼────────┼──────┤
│ Closed Fist │ 72%       │ 100%   │ 0.83 │  ✓ Found all, some false alarms
│ Finger Gun  │ 100%      │ 100%   │ 1.00 │  ✓✓ Perfect
│ Peace Sign  │ 100%      │ 100%   │ 1.00 │  ✓✓ Perfect
│ Thumbs Up   │ 100%      │ 89%    │ 0.94 │  ✓ No false alarms, missed 11%
└─────────────┴───────────┴────────┴──────┘

Correct Predictions: 1159 / 1246 (93.42%)
```

---

## 🎯 Why This Matters

### ✅ One Model, Four Gestures
- Before: Need 4 separate models/files
- After: Single model with 4 outputs

### ✅ Temporal Understanding
- Unlike static model, captures gesture motion
- 20-frame sequence provides context
- Better at distinguishing similar poses

### ✅ Efficient
- Smaller file size: 29 KB (vs ~116 KB for 4 binary models)
- Single forward pass for all 4 gestures
- Ready for Unity integration

### ✅ Better Insights
- Per-class performance visible
- See which gestures are confused
- Not just a single accuracy number

---

## 💡 Core Concepts to Remember

### 1. CrossEntropyLoss
- For: "Pick 1 class from N classes" problems
- Input: N logit values
- Output: Softmax probabilities (sum to 1.0)
- Loss: Penalizes wrong class predictions

### 2. Softmax Function
```
Converts [0.1, 2.0, 0.5, -0.1] → [0.01, 0.87, 0.10, 0.02]
         raw scores               probabilities (sum=1)
```

### 3. Argmax
```
Finds the index of maximum value
[0.01, 0.87, 0.10, 0.02] → index 1 (0.87 is largest)
```

### 4. LSTM for Sequences
```
Frame 1 → LSTM → hidden_state_1
Frame 2 + hidden_state_1 → LSTM → hidden_state_2
Frame 3 + hidden_state_2 → LSTM → hidden_state_3
...
Frame 20 + hidden_state_19 → LSTM → hidden_state_20

Use hidden_state_20 (has seen all frames) for prediction
```

---

## 📚 Where to Learn More

**In the full document:**
- **Section: "Detailed Code Changes"** - Line-by-line explanations
- **Section: "Key Concepts Explained"** - Deep dives into CrossEntropyLoss, Softmax, etc.
- **Section: "Detailed Comparison"** - Before/after code side-by-side
- **Section: "Training and Results"** - What the numbers mean

**External resources:**
- PyTorch docs: `torch.nn.CrossEntropyLoss`
- PyTorch docs: `torch.nn.LSTM`
- Fast.ai: Multiclass classification lesson
- CS231n: Understanding neural networks

---

## ✨ Summary of 9 Changes

1. ✅ Import statements - Removed unused libraries
2. ✅ Output dimension - 1 → 4
3. ✅ Loss function - BCEWithLogitsLoss → CrossEntropyLoss
4. ✅ Optimizer - SGD → Adam
5. ✅ Label extraction - Fixed to use integer, first frame
6. ✅ Training loop - Restructured for clarity
7. ✅ Prediction method - Threshold → Argmax
8. ✅ Evaluation metrics - Binary → Multiclass report
9. ✅ Model export - Fixed input shape for LSTM

---

## 🚀 Next Steps

**To understand this better:**
1. Read `MULTICLASS_DYNAMIC_IMPLEMENTATION.md`
2. Look at the actual code changes in `model_dynamic.py`
3. Run `process_dynamic_data.py` to see data generation
4. Modify hyperparameters and retrain to see effects

**To improve the model:**
- Increase sequence length (try 30, 40 frames)
- Add more LSTM layers
- Use bidirectional LSTM
- Implement early stopping
- Try different optimizers (RMSprop, etc.)
