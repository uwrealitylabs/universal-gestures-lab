import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import os
from . import utils
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Network aims to retrieve the N-dim fingerprint right before the model outputs to logits

#Hyperparameters
input_dim = 17 # assume same feature # as model.py for now
output_dim = 1 # binary classification (is feature or is not feature)
activation_threshold = 0.7 # threshold for classification as a feature

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "gestures_model_weights.json"

class feed_forward_gesture_mlp(nn.Module):
    def __init__(self, in_dim=input_dim, hidden_dim=128, out_dim=output_dim):
        super(feed_forward_gesture_mlp, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim) #input layer -> 128 dim hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2) # 128 dim -> 64 dim hidden layer
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim) # 64 dim layer -> 128 layer 
        self.output = nn.Linear(hidden_dim, out_dim) # final fc layer 128 -> 1 output dim 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return torch.sigmoid(x)



