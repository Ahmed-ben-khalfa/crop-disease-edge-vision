"""
Validation croisee (3 seeds) pour obtenir un intervalle de confiance
sur l'accuracy PlantDoc, plutot qu'un seul chiffre potentiellement instable.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

CATEGORY_MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust", "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot", "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy", "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight", "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy", "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew", "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight", "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy", "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight", "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy", "grape leaf black rot": "Grape___Black_rot",
}

def rebuild_groups(name_to_id):
    plantdoc = load_dataset("agyaatcoder/PlantDoc")
    groups, labels_check = [], []
    counter = 0
    for split_name in ["train", "test"]:
        for example in plantdoc[split_name]:
            objects = example["objects"]
            for bbox, category in zip(objects["bbox"], objects["category"]):
                if category not in CATEGORY_MAPPING:
                    continue
                mapped_name = CATEGORY_MAPPING[category]
                if mapped_name not in name_to_id:
                    continue
                groups.append(f"img_{counter}")
                labels_check.append(name_to_id[mapped_name])
            counter += 1
    return np.array(groups), np.array(labels_check)

def manifold_mixup(X, y, n_classes, n_per_class=150, alpha=0.4, noise_ratio=0.05, seed=42):
    synth_X, synth_y = [], []
    rng = np.random.default_rng(seed)
    gstd = X.std()
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        if len(idx_c) < 2:
            continue
        for _ in range(n_per_class):
            i, j = rng.choice(idx_c, size=2, replace=True)
            lam = rng.beta(alpha, alpha)
            mixed = lam * X[i] + (1 - lam) * X[j]
            mixed = mixed + rng.normal(0, gstd * noise_ratio, size=mixed.shape)
            synth_X.append(mixed)
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

def run_one_seed(seed, X_pd_full, y_pd_full, groups, X_pv_train, y_pv_train, X_val, y_val,
                  X_pv_test, y_pv_test, n_classes, class_weights_base):
    torch.manual_seed(seed)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    idx_train_full, idx_test = next(gss1.split(np.zeros(len(groups)), y_pd_full, groups=groups))
    groups_train_full = groups[idx_train_full]
    y_train_full = y_pd_full[idx_train_full]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed+1000)
    idx_tr2, idx_va2 = next(gss2.split(np.zeros(len(idx_train_full)), y_train_full, groups=groups_train_full))
    idx_train = idx_train_full[idx_tr2]
    idx_modelval = idx_train_full[idx_va2]

    X_pd_train, y_pd_train = X_pd_full[idx_train], y_pd_full[idx_train]
    X_pd_modelval, y_pd_modelval = X_pd_full[idx_modelval], y_pd_full[idx_modelval]
    X_pd_test, y_pd_test = X_pd_full[idx_test], y_pd_full[idx_test]

    X_synth, y_synth = manifold_mixup(X_pd_train, y_pd_train, n_classes, n_per_class=150, seed=seed)
    OVERSAMPLE_FACTOR = 3
    X_pd_boosted = np.concatenate([X_pd_train] * OVERSAMPLE_FACTOR + [X_synth], axis=0)
    y_pd_boosted = np.concatenate([y_pd_train] * OVERSAMPLE_FACTOR + [y_synth], axis=0)
    X_combined = np.concatenate([X_pv_train, X_pd_boosted], axis=0)
    y_combined = np.concatenate([y_pv_train, y_pd_boosted], axis=0)

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

    n_epochs = 30
    batch_size = 256
    n_samples = X_train_t.shape[0]
    best_val_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pdval_preds = torch.argmax(model(X_pdval_t), dim=1).numpy()
        pdval_acc = accuracy_score(y_pd_modelval, pdval_preds)
        if pdval_acc > best_val_acc:
            best_val_acc = pdval_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pd_test_preds = torch.argmax(model(torch.tensor(X_pd_test, dtype=torch.float32)), dim=1).numpy()
        pv_test_preds = torch.argmax(model(torch.tensor(X_pv_test, dtype=torch.float32)), dim=1).numpy()

    pd_acc = accuracy_score(y_pd_test, pd_test_preds)
    pd_f1 = f1_score(y_pd_test, pd_test_preds, average="macro")
    pv_acc = accuracy_score(y_pv_test, pv_test_preds)

    return pd_acc, pd_f1, pv_acc, len(idx_test)

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Reconstruction des groupes PlantDoc...")
    groups, labels_check = rebuild_groups(name_to_id)
    y_pd_full = np.load("data/plantdoc_labels.npy")
    X_pd_full = np.load("data/plantdoc_embeddings.npy")
    assert np.array_equal(labels_check, y_pd_full)

    X_pv_train = np.load("data/train_embeddings.npy")
    y_pv_train = np.load("data/train_labels.npy")
    X_val = np.load("data/val_embeddings.npy")
    y_val = np.load("data/val_labels.npy")
    X_pv_test = np.load("data/test_embeddings.npy")
    y_pv_test = np.load("data/test_labels.npy")

    seeds = [42, 7, 123]
    results = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"RUN {i}/3 (seed={seed})")
        print(f"{'='*60}")
        pd_acc, pd_f1, pv_acc, n_test = run_one_seed(
            seed, X_pd_full, y_pd_full, groups, X_pv_train, y_pv_train,
            X_val, y_val, X_pv_test, y_pv_test, n_classes, None
        )
        print(f"PlantDoc test acc={pd_acc:.4f} | f1={pd_f1:.4f} | (n={n_test}) | PlantVillage acc={pv_acc:.4f}")
        results.append((pd_acc, pd_f1, pv_acc))

    pd_accs = [r[0] for r in results]
    pd_f1s = [r[1] for r in results]

    print(f"\n{'='*60}")
    print("=== RESULTAT FINAL AVEC INTERVALLE DE CONFIANCE (3 seeds) ===")
    print(f"{'='*60}")
    print(f"PlantDoc accuracy: {np.mean(pd_accs)*100:.1f}% +/- {np.std(pd_accs)*100:.1f}% (min={min(pd_accs)*100:.1f}%, max={max(pd_accs)*100:.1f}%)")
    print(f"PlantDoc F1 macro: {np.mean(pd_f1s)*100:.1f}% +/- {np.std(pd_f1s)*100:.1f}%")
    print(f"\nCHIFFRE A CITER DANS LE RAPPORT: {np.mean(pd_accs)*100:.1f}% (+/- {np.std(pd_accs)*100:.1f}%), n=3 splits independants")

if __name__ == "__main__":
    main()
