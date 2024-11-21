import numpy as np
from sklearn.metrics import confusion_matrix
def compute_confusion_matrix(y_actual, y_predicted):
    cm = confusion_matrix(y_actual, y_predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}