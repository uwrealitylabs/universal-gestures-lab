import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

output_dim = 1  # binary classification for thumbs up or down
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for classification as a thumbs up

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "model_weights.json"


# Model
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.sigmoid(out)
        # Linear function (readout)
        out = self.fc2(out)
        return torch.sigmoid(out)


# Data set
def split_feature_label(data):
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X, self.Y = split_feature_label(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Loader fn
def load_data(dataset, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Load environment variables
load_dotenv()

def log_training_metrics(auc_pr, auc_roc, final_loss, model_type):
    """Logs training metrics to the NeonDB database."""
    conn = None
    try:
        db_url = os.getenv("NEONDB_URL")
        conn = psycopg2.connect(db_url, sslmode='require')
        cursor = conn.cursor()

        # Ensure data types are Python-native
        auc_pr = float(auc_pr)  # Convert to native float
        auc_roc = float(auc_roc)  # Convert to native float
        final_loss = float(final_loss)  # Convert to native float

        '''
        # Define the maximum number of rows allowed in the table
        MAX_ROWS = 1000000  # Adjust this value based on the NeonDB capacity

        # Check the current number of rows in the table
        cursor.execute("SELECT COUNT(*) FROM public.model_tracking;")
        row_count = cursor.fetchone()[0]

        if row_count >= MAX_ROWS:
            # Number of oldest rows to delete to free up space
            N = 100  # Adjust this value as needed

            # Delete the oldest N rows based on the datetime column
            delete_query = """
            DELETE FROM public.model_tracking
            WHERE ctid IN (
                SELECT ctid FROM public.model_tracking
                ORDER BY datetime ASC
                LIMIT %s
            );
            """
            cursor.execute(delete_query, (N,))
            conn.commit()
            print(f"Deleted {N} oldest rows to free up space.")
        '''

        query = """
        INSERT INTO public.model_tracking (datetime, auc_pr, auc_roc, final_loss, model_type)
        VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(query, (datetime.now(), auc_pr, auc_roc, final_loss, model_type))
        conn.commit()
        print("Training metrics logged successfully.")

    except psycopg2.Error as e:
        # Check if the error is due to storage capacity
        if 'could not extend file' in str(e).lower() or 'no space left on device' in str(e).lower():
            print("Storage capacity reached. Deleting oldest entries to free up space.")
            try:
                # Number of oldest rows to delete
                N = 100  # Adjust as needed

                # Delete the oldest N rows based on the datetime column
                delete_query = """
                DELETE FROM public.model_tracking
                WHERE ctid IN (
                    SELECT ctid FROM public.model_tracking
                    ORDER BY datetime ASC
                    LIMIT %s
                );
                """
                cursor.execute(delete_query, (N,))
                conn.commit()
                print(f"Deleted {N} oldest rows to free up space.")

                # Retry the insertion
                cursor.execute(query, (datetime.now(), auc_pr, auc_roc, final_loss, model_type))
                conn.commit()
                print("Training metrics logged successfully after freeing up space.")
            except Exception as del_e:
                print("Error during deletion or reinsertion:", del_e)
        else:
            print("Error logging training metrics:", e)

    except Exception as e:
        print("Error logging training metrics:", e)
    finally:
        if conn:
            cursor.close()
            conn.close()

def main():
    train_path = "train_data/train_0.pt"
    test_path = "test_data/test_0.pt"
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    batch_size = 64
    n_iters = len(train_data) * 5  # 5 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))

    X_train = torch.tensor(train_data[:, :-1])
    y_train = torch.tensor(train_data[:, -1])
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=16
    )

    X_test = torch.tensor(test_data[:, :-1])
    y_test = torch.tensor(test_data[:, -1])
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=True, batch_size=16
    )

    model = FeedforwardNeuralNetModel(input_dim, 100, output_dim)
    criterion = nn.BCELoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    iter = 0

    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            Y = Y.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, Y.float())
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                correct = 0
                total = 0
                all_labels = []
                all_probs = []
                for X, Y in test_loader:
                    outputs = model(X.float())
                    probs = outputs.detach().numpy().flatten()
                    predicted = (outputs > detect_threshold).float()
                    total += Y.size(0)
                    correct += (predicted == Y.view(-1, 1)).sum().item()
                    all_labels.extend(Y.numpy())
                    all_probs.extend(probs)

                accuracy = 100 * correct / total
                auc_roc = roc_auc_score(all_labels, all_probs)
                precision, recall, _ = precision_recall_curve(all_labels, all_probs)
                auc_pr = auc(recall, precision)

                # Example: Log metrics to the database
                model_type = "binary classification"  # Adjust based on your specific model type
                log_training_metrics(auc_pr, auc_roc, loss.item(), model_type)

                print(
                    "Iteration: {}. Loss: {}. Accuracy: {}. AUC-ROC: {:.4f}. AUC-PR: {:.4f}".format(
                        iter, loss.item(), accuracy, auc_roc, auc_pr
                    )
                )

    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    # Store as onnx for compatibility with Unity Barracuda
    onnx_program = torch.onnx.dynamo_export(model, torch.randn(1, input_dim))
    onnx_program.save(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME.split(".")[0] + ".onnx")

    print("\n--- Model Training Complete ---")
    print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)


if __name__ == "__main__":
    main()
