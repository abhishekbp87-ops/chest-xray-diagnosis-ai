# evaluate.py
import torch
import numpy as np
import os
from src.medical_model import MedicalNet
from src.data_processor import make_loaders
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dl, val_dl, test_dl, classes = make_loaders("data/chest_xray_data.csv", batch_size=32, img_size=224)

model = MedicalNet(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

y_true, y_scores = [], []
with torch.no_grad():
    for x, y in test_dl:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_scores.extend(probs[:, 1].tolist())   # probability for class 1 (PNEUMONIA)
        y_true.extend(y.numpy().tolist())

y_pred = [1 if p > 0.5 else 0 for p in y_scores]
print(classification_report(y_true, y_pred, target_names=classes))

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/roc.png")
plt.close()

print("Saved results in results/")