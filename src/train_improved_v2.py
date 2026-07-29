"""
Version corrigee: suppression de la fuite de donnees via selection de modele.
Le test set PlantDoc n'est utilise QU'UNE SEULE FOIS, a la toute fin.
Correction methodologique majeure.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def manifold_mixup(X, y, n_classes, n_synthetic_per_class=150, alpha=0.4, noise_std_ratio=0.05):
    synth_X, synth_y = [], []
    rng = np.random.default_rng(42)
    global_std = X.std()
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        if len(idx_c) < 2:
            continue
        for _ in range(n_synthetic_per_class):
            i, j = rng.choice(idx_c, size=2, replace=True)
            lam = rng.beta(alpha, alpha)
            mixed = lam * X[i] + (1 - lam) * X[j]
            noise = rng.normal(0, global_std * noise_std_ratio, size=mixed.shape)
            synth_X.append(mixed + noise)
            synth_y.append(c)
    return np.array(synth_X, dtype=np.float32), np.array(synth_y)

class MLPHead(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)

def main():
    print("Chargement des embeddings...")
    X_pv_train = np.load("data/train_embeddings.npy")
    y_pv_train = np.load("data/train_labels.npy")
    X_val = np.load("data/val_embeddings.npy")
    y_val = np.load("data/val_labels.npy")
    X_pd_full = np.load("data/plantdoc_embeddings.npy")
    y_pd_full = np.load("data/plantdoc_labels.npy")
    X_pd_test = np.load("data/plantdoc_test_embeddings.npy")   # JAMAIS touche avant la fin
    y_pd_test = np.load("data/plantdoc_test_labels.npy")
    X_pv_test = np.load("data/test_embeddings.npy")
    y_pv_test = np.load("data/test_labels.npy")

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]

    # Meme split PlantDoc train(70%)/test(30%) que l'original (seed=42, deja isole)
    idx_all = np.arange(len(y_pd_full))
    idx_train_pd, _ = train_test_split(idx_all, test_size=0.3, random_state=42, stratify=y_pd_full)
    X_pd_train_full, y_pd_train_full = X_pd_full[idx_train_pd], y_pd_full[idx_train_pd]

    # NOUVEAU: on decoupe encore ce train PlantDoc en train/val (80/20)
    # pour choisir le meilleur epoch SANS jamais toucher au vrai test
    idx_pd_train2 = np.arange(len(y_pd_train_full))
    idx_pd_tr, idx_pd_va = train_test_split(idx_pd_train2, test_size=0.2, random_state=123, stratify=y_pd_train_full)
    X_pd_train, y_pd_train = X_pd_train_full[idx_pd_tr], y_pd_train_full[idx_pd_tr]
    X_pd_modelval, y_pd_modelval = X_pd_train_full[idx_pd_va], y_pd_train_full[idx_pd_va]

    print(f"PlantVillage train: {len(y_pv_train)}")
    print(f"PlantDoc train (pour apprendre): {len(y_pd_train)}")
    print(f"PlantDoc val (pour choisir le meilleur epoch, PAS le test): {len(y_pd_modelval)}")
    print(f"PlantDoc test (jamais touche avant la toute fin): {len(y_pd_test)}")

    print("\nGeneration d'embeddings synthetiques (manifold mixup) sur PlantDoc TRAIN uniquement...")
    X_synth, y_synth = manifold_mixup(X_pd_train, y_pd_train, n_classes, n_synthetic_per_class=150)
    print(f"Embeddings synthetiques generes: {len(y_synth)}")

    OVERSAMPLE_FACTOR = 3
    X_pd_boosted = np.concatenate([X_pd_train] * OVERSAMPLE_FACTOR + [X_synth], axis=0)
    y_pd_boosted = np.concatenate([y_pd_train] * OVERSAMPLE_FACTOR + [y_synth], axis=0)

    X_combined = np.concatenate([X_pv_train, X_pd_boosted], axis=0)
    y_combined = np.concatenate([y_pv_train, y_pd_boosted], axis=0)
    print(f"Dataset final d'entrainement: {len(y_combined)} images")

    from collections import Counter
    counts = Counter(y_combined.tolist())
    total = len(y_combined)
    class_weights = np.array([total / (n_classes * counts.get(i, 1)) for i in range(n_classes)], dtype=np.float32)
    weights_tensor = torch.tensor(class_weights)

    X_train_t = torch.tensor(X_combined, dtype=torch.float32)
    y_train_t = torch.tensor(y_combined, dtype=torch.long)
    X_pdval_t = torch.tensor(X_pd_modelval, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    embedding_dim = X_combined.shape[1]
    model = MLPHead(embedding_dim, n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)

    n_epochs = 40
    batch_size = 256
    n_samples = X_train_t.shape[0]
    best_pd_modelval_acc = 0.0

    print(f"\nEntrainement ({n_epochs} epochs)... selection du modele sur PlantDoc VAL (pas test)")
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_preds = torch.argmax(model(X_val_t), dim=1).numpy()
            pdval_preds = torch.argmax(model(X_pdval_t), dim=1).numpy()
        val_f1 = f1_score(y_val, val_preds, average="macro")
        pdval_acc = accuracy_score(y_pd_modelval, pdval_preds)

        print(f"Epoch {epoch+1:2d}/{n_epochs} | loss={avg_loss:.4f} | val_f1(PV)={val_f1:.4f} | val_acc(PlantDoc, PAS test)={pdval_acc:.4f}")

        if pdval_acc > best_pd_modelval_acc:
            best_pd_modelval_acc = pdval_acc
            torch.save(model.state_dict(), "models/mlp_v2_best.pt")

    print(f"\nMeilleur modele (selectionne sur PlantDoc VAL): models/mlp_v2_best.pt")

    # --- EVALUATION FINALE UNIQUE sur le vrai test set, jamais vu avant ---
    model.load_state_dict(torch.load("models/mlp_v2_best.pt"))
    model.eval()
    with torch.no_grad():
        pd_test_preds = torch.argmax(model(torch.tensor(X_pd_test, dtype=torch.float32)), dim=1).numpy()
        pv_test_preds = torch.argmax(model(torch.tensor(X_pv_test, dtype=torch.float32)), dim=1).numpy()

    pd_test_acc = accuracy_score(y_pd_test, pd_test_preds)
    pd_test_f1 = f1_score(y_pd_test, pd_test_preds, average="macro")
    pv_test_acc = accuracy_score(y_pv_test, pv_test_preds)
    pv_test_f1 = f1_score(y_pv_test, pv_test_preds, average="macro")

    print("\n" + "="*70)
    print("=== RESULTAT FINAL HONNETE (test set jamais utilise avant maintenant) ===")
    print("="*70)
    print(f"AVANT correction methodologique (biaise, selection sur test): PlantDoc acc=0.8040")
    print(f"APRES correction (selection sur val, evaluation propre sur test): PlantDoc acc={pd_test_acc:.4f}, f1={pd_test_f1:.4f}")
    print(f"\nEcart du a la fuite corrigee: {(0.8040 - pd_test_acc)*100:+.1f} points (l'ancien chiffre etait optimiste de ce montant)")
    print(f"\nPlantVillage test: acc={pv_test_acc:.4f}, f1={pv_test_f1:.4f}")

    # --- Analyse d'erreurs sur le VRAI test set ---
    cm = confusion_matrix(y_pd_test, pd_test_preds, labels=list(range(n_classes)))
    errors = []
    for true_c in range(n_classes):
        for pred_c in range(n_classes):
            if true_c != pred_c and cm[true_c, pred_c] > 0:
                errors.append((cm[true_c, pred_c], labels_names[true_c], labels_names[pred_c]))
    errors.sort(reverse=True)

    print("\n=== Top 10 confusions (evaluation propre sur test set) ===")
    for count, true_name, pred_name in errors[:10]:
        print(f"  {true_name:<45} -> {pred_name:<45} : {count} erreurs")

if __name__ == "__main__":
    main()
