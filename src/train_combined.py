"""
Split PlantDoc (train/test) + fusion avec PlantVillage + reentrainement.
Semaine 3 - Etape 2 : Correction du domain gap.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

def main():
    print("Chargement des embeddings existants...")
    X_pv_train = np.load("data/train_embeddings.npy")
    y_pv_train = np.load("data/train_labels.npy")
    X_pv_val = np.load("data/val_embeddings.npy")
    y_pv_val = np.load("data/val_labels.npy")

    X_pd = np.load("data/plantdoc_embeddings.npy")
    y_pd = np.load("data/plantdoc_labels.npy")

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]

    # --- Split PlantDoc: 70% train (a fusionner), 30% test (jamais vu, evaluation finale) ---
    print(f"\nSplit de PlantDoc ({len(y_pd)} images) en 70% train / 30% test...")
    idx_all = np.arange(len(y_pd))
    idx_train_pd, idx_test_pd = train_test_split(
        idx_all, test_size=0.3, random_state=42, stratify=y_pd
    )
    X_pd_train, y_pd_train = X_pd[idx_train_pd], y_pd[idx_train_pd]
    X_pd_test, y_pd_test = X_pd[idx_test_pd], y_pd[idx_test_pd]

    print(f"PlantDoc train (a fusionner): {len(y_pd_train)}")
    print(f"PlantDoc test (reserve, jamais utilise en entrainement): {len(y_pd_test)}")

    np.save("data/plantdoc_test_embeddings.npy", X_pd_test)
    np.save("data/plantdoc_test_labels.npy", y_pd_test)

    # --- Fusion: PlantVillage train + PlantDoc train ---
    X_combined = np.concatenate([X_pv_train, X_pd_train], axis=0)
    y_combined = np.concatenate([y_pv_train, y_pd_train], axis=0)
    print(f"\nDataset combine pour l'entrainement: {len(y_combined)} images")
    print(f"  (PlantVillage: {len(y_pv_train)} + PlantDoc: {len(y_pd_train)})")

    # --- Recalcul des poids de classe sur le dataset combine ---
    from collections import Counter
    counts = Counter(y_combined.tolist())
    total = len(y_combined)
    class_weights = np.array([
        total / (n_classes * counts.get(i, 1)) for i in range(n_classes)
    ], dtype=np.float32)
    weights_tensor = torch.tensor(class_weights)

    # --- Entrainement ---
    X_train_t = torch.tensor(X_combined, dtype=torch.float32)
    y_train_t = torch.tensor(y_combined, dtype=torch.long)
    X_val_t = torch.tensor(X_pv_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_pv_val, dtype=torch.long)

    embedding_dim = X_combined.shape[1]
    model = nn.Linear(embedding_dim, n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    n_epochs = 30
    batch_size = 256
    n_samples = X_train_t.shape[0]
    best_val_f1 = 0.0

    print(f"\nReentrainement sur dataset combine ({n_epochs} epochs)...")
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

        model.eval()
        with torch.no_grad():
            val_preds = torch.argmax(model(X_val_t), dim=1).numpy()
        val_acc = accuracy_score(y_pv_val, val_preds)
        val_f1 = f1_score(y_pv_val, val_preds, average="macro")

        print(f"Epoch {epoch+1:2d}/{n_epochs} | loss={avg_loss:.4f} | val_acc(PV)={val_acc:.4f} | val_f1(PV)={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "models/linear_probe_combined_best.pt")

    print(f"\nMeilleur modele sauvegarde: models/linear_probe_combined_best.pt")

    # --- EVALUATION FINALE: comparaison avant/apres sur PlantDoc test (jamais vu) ---
    model.load_state_dict(torch.load("models/linear_probe_combined_best.pt"))
    model.eval()

    X_pd_test_t = torch.tensor(X_pd_test, dtype=torch.float32)
    with torch.no_grad():
        pd_preds = torch.argmax(model(X_pd_test_t), dim=1).numpy()

    pd_acc = accuracy_score(y_pd_test, pd_preds)
    pd_f1 = f1_score(y_pd_test, pd_preds, average="macro")

    print("\n" + "="*60)
    print("=== COMPARAISON AVANT / APRES CORRECTION DU DOMAIN GAP ===")
    print("="*60)
    print(f"AVANT (baseline PlantVillage only) sur PlantDoc: accuracy=0.3746, f1_macro=0.2586")
    print(f"APRES (PlantVillage + PlantDoc mixe) sur PlantDoc test: accuracy={pd_acc:.4f}, f1_macro={pd_f1:.4f}")
    print(f"\nAmelioration: +{(pd_acc - 0.3746)*100:.1f} points d'accuracy")
    print(f"Amelioration F1 macro: +{(pd_f1 - 0.2586)*100:.1f} points")

    # Verification qu'on n'a pas degrade PlantVillage
    X_pv_test = np.load("data/test_embeddings.npy")
    y_pv_test = np.load("data/test_labels.npy")
    X_pv_test_t = torch.tensor(X_pv_test, dtype=torch.float32)
    with torch.no_grad():
        pv_preds = torch.argmax(model(X_pv_test_t), dim=1).numpy()
    pv_acc = accuracy_score(y_pv_test, pv_preds)
    pv_f1 = f1_score(y_pv_test, pv_preds, average="macro")
    print(f"\nVerification PlantVillage test (ne doit pas trop chuter): accuracy={pv_acc:.4f}, f1_macro={pv_f1:.4f}")
    print(f"(Rappel avant fusion: accuracy=0.9876, f1_macro=0.9838)")

if __name__ == "__main__":
    main()
