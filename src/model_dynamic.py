import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dim = 1  # binary classification
input_dim = 17  # 17 features per timestep
detect_threshold = 0.7

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "trained_model")
SAVE_MODEL_FILENAME = "model_dynamic_weights.json"
TRAIN_PATH = os.path.join(SCRIPT_DIR, "train_data", "train_sequences_0.pt")
TEST_PATH = os.path.join(SCRIPT_DIR, "test_data", "test_sequences_0.pt")
NORM_STATS_PATH = os.path.join(SAVE_MODEL_PATH, "normalization_stats.json")

class FixedNormalization(nn.Module):
    """Fixed normalization layer with pre-computed statistics"""
    def __init__(self, mean, std):
        super(FixedNormalization, self).__init__()
        # Register as buffers (non-trainable, but saved with model)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
    
    def forward(self, x):
        # For 4D input (batch, height, width, channels)
        # Normalize the channels dimension
        if x.dim() == 4:
            # mean and std should be applied to the channels dimension
            mean = self.mean.view(1, 1, 1, -1)
            std = self.std.view(1, 1, 1, -1)
            x = (x - mean) / std
        elif x.dim() == 3:
            # For 3D input (batch, seq, features)
            x = (x - self.mean) / self.std
        return x

class Conv2D_Model(nn.Module):
    def __init__(self, input_dim, output_dim, sequence_length=15, norm_mean=None, norm_std=None):
        super(Conv2D_Model, self).__init__()
        
        self.sequence_length = sequence_length
        
        # Add normalization layer if statistics provided
        if norm_mean is not None and norm_std is not None:
            self.normalize = FixedNormalization(norm_mean, norm_std)
        else:
            self.normalize = None
        
        # Use Conv2D maintains NHWC format
        # Input: (batch, height=seq, width=1, channels=features)
        # Conv2D processes this naturally without needing transpose
        
        # First we need to convert from NHWC to NCHW for PyTorch Conv2D
        # But we'll handle this with permute operations that Unity can understand
        
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Expected input: (batch, height=seq, width=1, channels=features)
        # This is NHWC format: (N, H, W, C) = (1, 15, 1, 17)
        
        # Handle different input formats
        if x.dim() == 3:
            # If 3D (batch, seq, features), add width dimension
            x = x.unsqueeze(2)  # (batch, seq, 1, features)
        
        # Ensure we have 4D tensor
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D with shape {x.shape}")
        
        # Apply normalization in NHWC format
        if self.normalize is not None:
            x = self.normalize(x)
        
        # Convert from NHWC to NCHW for PyTorch Conv2D
        # (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Apply Conv2D layers (now in NCHW format)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)
        
        # Final classification
        x = self.dropout(x)
        out = self.fc(x)
        
        return out

def split_feature_label(data, labels, lengths):
    """Split features and labels for dynamic sequence data"""
    X = data  # All features
    Y = labels  # One label per sequence
    return X, Y, lengths

def main():
    print("Training Conv2D model with NHWC format...")
    
    # Load data
    train_loaded = torch.load(TRAIN_PATH, weights_only=False)
    test_loaded = torch.load(TEST_PATH, weights_only=False)
    
    train_data, train_labels, train_lengths = train_loaded['data'], train_loaded['labels'], train_loaded['lengths']
    test_data, test_labels, test_lengths = test_loaded['data'], test_loaded['labels'], test_loaded['lengths']
    
    # Load normalization statistics if available
    norm_mean, norm_std = None, None
    if os.path.exists(NORM_STATS_PATH):
        print(f"Loading normalization statistics from {NORM_STATS_PATH}")
        with open(NORM_STATS_PATH, 'r') as f:
            norm_stats = json.load(f)
            norm_mean = norm_stats['mean']
            norm_std = norm_stats['std']
        print("✅ Normalization will be embedded in the model")
    else:
        print("⚠️  WARNING: No normalization statistics found. Training without normalization.")
    
    batch_size = 64
    n_iters = len(train_data) * 10  # 10 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))
    
    X_train, y_train, lengths_train = split_feature_label(train_data, train_labels, train_lengths)
    X_test, y_test, lengths_test = split_feature_label(test_data, test_labels, test_lengths)
    
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train, lengths_train)), shuffle=True, batch_size=batch_size
    )
    
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test, lengths_test)), shuffle=True, batch_size=batch_size
    )
    
    # Create model with normalization embedded
    model = Conv2D_Model(input_dim, output_dim, norm_mean=norm_mean, norm_std=norm_std).to(device)
    
    # Print model info
    if norm_mean is not None:
        print(f"\nNormalization embedded in model:")
        print(f"  Mean: {[f'{m:.4f}' for m in norm_mean[:5]]}... (showing first 5)")
        print(f"  Std:  {[f'{s:.4f}' for s in norm_std[:5]]}... (showing first 5)")
    
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (X, Y, lengths) in enumerate(train_loader):
            model.train()
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            Y = Y.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, Y.float())
            loss.backward()
            optimizer.step()
            iter += 1
            
            if iter % 100 == 0:
                correct = 0
                total = 0
                all_labels = []
                all_probs = []
                
                model.eval()
                with torch.inference_mode():
                    for X, Y, lengths in test_loader:
                        X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
                        outputs = model(X.float())
                        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                        predicted = (torch.sigmoid(outputs) > detect_threshold).float()
                        total += Y.size(0)
                        correct += (predicted == Y.view(-1, 1)).sum().item()
                        all_labels.extend(Y.cpu().numpy())
                        all_probs.extend(probs)
                    
                    accuracy = 100 * correct / total
                    if len(set(all_labels)) > 1:
                        auc_roc = roc_auc_score(all_labels, all_probs)
                        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
                        auc_pr = auc(recall, precision)
                        print(f"Iteration: {iter}. Loss: {loss.item():.4f}. Accuracy: {accuracy:.2f}. AUC-ROC: {auc_roc:.4f}. AUC-PR: {auc_pr:.4f}")
                    else:
                        print(f"Iteration: {iter}. Loss: {loss.item():.4f}. Accuracy: {accuracy:.2f}")
    
    # Save model weights
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}
    
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    with open(os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME), "w") as f:
        json.dump(serializable_state_dict, f)
    
    print("\nExporting ONNX model with Unity-compatible NHWC format...")
    
    # Create dummy 4D NHWC input(batch=1, height=15, width=1, channels=17)
    dummy_input = torch.randn(1, 15, 1, input_dim).to(device)
    
    print(f"Exporting with 4D NHWC shape: {dummy_input.shape}")
    print("Format: (batch=1, height=15, width=1, channels=17)")
    print("Unity will expand this to 8D: [1,1,1,1,1,15,1,17]")
    print("Dimensions will remain in correct positions throughout processing")
    
    # Set model to eval mode for export
    model.eval()
    
    # Export model with opset 9 for Unity compatibility
    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME.split(".")[0] + ".onnx"),
        input_names=['input'],
        output_names=['output'],
        opset_version=9,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\nModel saved to: {os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME)}")
    print(f"ONNX model saved to: {os.path.join(SAVE_MODEL_PATH, SAVE_MODEL_FILENAME.split('.')[0] + '.onnx')}")

if __name__ == "__main__":
    main()