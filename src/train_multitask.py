"""
Mapping severite (proxy expert) + entrainement multi-tache (maladie + severite).
Semaine 4 - Etape 1 : Tete de severite.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

with open("data/split_info.json", "r") as f:
    split_info = json.load(f)
labels_names = split_info["labels_names"]

SEVERITY_MAPPING = {
    "Apple___Apple_scab": 1, "Apple___Black_rot": 2, "Apple___Cedar_apple_rust": 1, "Apple___healthy": 0,
    "Blueberry___healthy": 0,
    "Cherry_(including_sour)___Powdery_mildew": 1, "Cherry_(including_sour)___healthy": 0,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 1, "Corn_(maize)___Common_rust_": 1,
    "Corn_(maize)___Northern_Leaf_Blight": 2, "Corn_(maize)___healthy": 0,
    "Grape___Black_rot": 2, "Grape___Esca_(Black_Measles)": 2,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 1, "Grape___healthy": 0,
    "Orange___Haunglongbing_(Citrus_greening)": 2,
    "Peach___Bacterial_spot": 1, "Peach___healthy": 0,
    "Pepper,_bell___Bacterial_spot": 1, "Pepper,_bell___healthy": 0,
    "Potato___Early_blight": 1, "Potato___Late_blight": 2, "Potato___healthy": 0,
    "Raspberry___healthy": 0,
    "Soybean___healthy": 0,
    "Squash___Powdery_mildew": 1,
    "Strawberry___Leaf_scorch": 1, "Strawberry___healthy": 0,
    "Tomato___Bacterial_spot": 1, "Tomato___Early_blight": 1, "Tomato___Late_blight": 2,
    "Tomato___Leaf_Mold": 1, "Tomato___Septoria_leaf_spot": 1,
    "Tomato___Spider_mites Two-spotted_spider_mite": 1, "Tomato___Target_Spot": 1,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 2, "Tomato___Tomato_mosaic_virus": 2,
    "Tomato___healthy": 0,
}

SEVERITY_NAMES = ["Sain", "Leger", "Severe"]

def labels_to_severity(labels_array, labels_names):
    return np.array([SEVERITY_MAPPING[labels_names[l]] for l in labels_array])

class MultiTaskHead(nn.Module):
    def __init__(self, embedding_dim, n_disease_classes, n_severity_classes=3):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.disease_head = nn.Linear(256, n_disease_classes)
        self.severity_head = nn.Linear(256, n_severity_classes)

    def forward(self, x):
        shared_features = self.shared(x)
        return self.disease_head(shared_features), self.severity_head(shared_features)

def main():
    print("Chargement des embeddings (dataset combine PlantVillage+PlantDoc)...")
    X_pv_train = np.load("data/train_embeddings.npy")
    y_pv_train = np.load("data/train_labels.npy")
    X_pd_train_full = np.load("data/plantdoc_embeddings.npy")
    y_pd_train_full = np.load("data/plantdoc_labels.npy")

    X_val = np.load("data/val_embeddings.npy")
    y_val = np.load("data/val_labels.npy")

    X_train = np.concatenate([X_pv_train, X_pd_train_full], axis=0)
    y_train = np.concatenate([y_pv_train, y_pd_train_full], axis=0)

    sev_train = labels_to_severity(y_train, labels_names)
    sev_val = labels_to_severity(y_val, labels_names)

    print(f"Train: {len(y_train)} images | Distribution severite: {np.bincount(sev_train)}")

    n_classes = split_info["n_classes"]
    embedding_dim = X_train.shape[1]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    sev_train_t = torch.tensor(sev_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    sev_val_t = torch.tensor(sev_val, dtype=torch.long)

    model = MultiTaskHead(embedding_dim, n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_disease = nn.CrossEntropyLoss()
    criterion_severity = nn.CrossEntropyLoss()
    LAMBDA_SEVERITY = 0.5

    n_epochs = 30
    batch_size = 256
    n_samples = X_train_t.shape[0]
    best_combined_score = 0.0

    print(f"\nEntrainement multi-tache ({n_epochs} epochs)...")
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb, sb = X_train_t[idx], y_train_t[idx], sev_train_t[idx]
            optimizer.zero_grad()
            disease_logits, severity_logits = model(xb)
            loss_disease = criterion_disease(disease_logits, yb)
            loss_severity = criterion_severity(severity_logits, sb)
            loss = loss_disease + LAMBDA_SEVERITY * loss_severity
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / n_samples

        model.eval()
        with torch.no_grad():
            disease_logits, severity_logits = model(X_val_t)
            disease_preds = torch.argmax(disease_logits, dim=1).numpy()
            severity_preds = torch.argmax(severity_logits, dim=1).numpy()

        disease_f1 = f1_score(y_val, disease_preds, average="macro")
        severity_acc = accuracy_score(sev_val, severity_preds)
        severity_qwk = cohen_kappa_score(sev_val, severity_preds, weights="quadratic")

        combined_score = disease_f1 + severity_qwk
        print(f"Epoch {epoch+1:2d}/{n_epochs} | loss={avg_loss:.4f} | disease_f1={disease_f1:.4f} | severity_acc={severity_acc:.4f} | severity_QWK={severity_qwk:.4f}")

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save(model.state_dict(), "models/multitask_best.pt")

    print(f"\nMeilleur modele sauvegarde: models/multitask_best.pt")
    print(f"\n=== RESUME FINAL ===")
    print(f"Disease F1 macro: {disease_f1:.4f}")
    print(f"Severity accuracy: {severity_acc:.4f}")
    print(f"Severity QWK (Quadratic Weighted Kappa): {severity_qwk:.4f}")
    print("(QWK > 0.6 = accord substantiel, > 0.8 = accord quasi-parfait - standard en classification ordinale medicale)")

if __name__ == "__main__":
    main()
