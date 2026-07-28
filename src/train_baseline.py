"""
Entrainement du classifieur lineaire (linear probing) sur les embeddings DINOv2.
Semaine 2 - Etape 2 : Baseline model.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    print("Chargement des embeddings...")
    X_train = np.load("data/train_embeddings.npy")
    y_train = np.load("data/train_labels.npy")
    X_val = np.load("data/val_embeddings.npy")
    y_val = np.load("data/val_labels.npy")

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Poids de classe -> tensor pour la loss
    class_weights_dict = split_info["class_weights"]
    weights_array = np.array([class_weights_dict[str(i)] for i in range(n_classes)], dtype=np.float32)
    weights_tensor = torch.tensor(weights_array)

    # Conversion en tensors PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Modele: simple couche lineaire (384 -> 38 classes)
    embedding_dim = X_train.shape[1]
    model = nn.Linear(embedding_dim, n_classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    n_epochs = 30
    batch_size = 256
    n_samples = X_train_t.shape[0]

    print(f"\nDebut de l'entrainement ({n_epochs} epochs)...")
    best_val_f1 = 0.0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / n_samples

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = torch.argmax(val_logits, dim=1).numpy()

        val_acc = accuracy_score(y_val, val_preds)
        val_f1_macro = f1_score(y_val, val_preds, average="macro")

        print(f"Epoch {epoch+1:2d}/{n_epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1_macro={val_f1_macro:.4f}")

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            torch.save(model.state_dict(), "models/linear_probe_best.pt")

    print(f"\nMeilleur F1 macro (val): {best_val_f1:.4f}")
    print("Modele sauvegarde: models/linear_probe_best.pt")

    # Rapport detaille final avec le meilleur modele
    model.load_state_dict(torch.load("models/linear_probe_best.pt"))
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_preds = torch.argmax(val_logits, dim=1).numpy()

    print("\n=== RAPPORT DE CLASSIFICATION (VAL) ===")
    print(classification_report(y_val, val_preds, target_names=labels_names, zero_division=0))

if __name__ == "__main__":
    main()
