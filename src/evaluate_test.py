"""
Evaluation finale du baseline sur le test set.
Semaine 2 - Etape 3 : Validation du pipeline complet.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    X_test = np.load("data/test_embeddings.npy")
    y_test = np.load("data/test_labels.npy")

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]
    embedding_dim = X_test.shape[1]

    model = nn.Linear(embedding_dim, n_classes)
    model.load_state_dict(torch.load("models/linear_probe_best.pt"))
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")

    print(f"=== RESULTATS TEST SET (jamais vu) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"\nNombre d'images test: {len(y_test)}")

if __name__ == "__main__":
    main()
