"""
Semaine 7 (amelioration) - Reentrainement avec augmentation de corruptions.
Objectif: rendre le modele robuste au flou/bruit/JPEG identifies comme fragiles.
On expose le modele a des versions corrompues des images d'entrainement PlantDoc,
avec le meme label, pour qu'il apprenne l'invariance a ces perturbations.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import io
from PIL import Image, ImageFilter, ImageEnhance
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

MODEL_NAME = "facebook/dinov2-small"

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

def corrupt_blur(img, severity):
    radius = [1, 2.5, 5][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def corrupt_noise(img, severity):
    arr = np.array(img).astype(np.float32)
    std = [10, 25, 50][severity - 1]
    noise = np.random.default_rng(0).normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def corrupt_jpeg(img, severity):
    quality = [40, 15, 5][severity - 1]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

CORRUPTION_FNS = [corrupt_blur, corrupt_noise, corrupt_jpeg]  # les 3 plus dommageables identifiees

def rebuild_crops(name_to_id, indices):
    plantdoc = load_dataset("agyaatcoder/PlantDoc")
    all_crops, all_labels = [], []
    for split_name in ["train", "test"]:
        for example in plantdoc[split_name]:
            img = example["image"].convert("RGB")
            objects = example["objects"]
            for bbox, category in zip(objects["bbox"], objects["category"]):
                if category not in CATEGORY_MAPPING:
                    continue
                mapped_name = CATEGORY_MAPPING[category]
                if mapped_name not in name_to_id:
                    continue
                x, y, w, h = bbox
                crop = img.crop((x, y, x + w, y + h))
                all_crops.append(crop)
                all_labels.append(name_to_id[mapped_name])
    y_check = np.load("data/plantdoc_labels.npy")
    assert np.array_equal(np.array(all_labels), y_check), "INCOHERENCE - arret"
    sel_crops = [all_crops[i] for i in indices]
    sel_labels = [all_labels[i] for i in indices]
    return sel_crops, sel_labels

def extract_embeddings(model, processor, images, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Chargement de DINOv2...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dino = AutoModel.from_pretrained(MODEL_NAME)
    dino.eval()

    print("Reconstruction des crops PlantDoc TRAIN (indices de production)...")
    train_idx = np.load("data/production_train_idx.npy")
    train_crops, train_labels = rebuild_crops(name_to_id, train_idx)
    print(f"PlantDoc train: {len(train_crops)} images")

    # --- Generation de versions corrompues (aleatoire, severite variable) ---
    print("\nGeneration de versions corrompues (flou/bruit/JPEG) pour l'augmentation robuste...")
    rng = np.random.default_rng(42)
    corrupted_crops = []
    corrupted_labels = []
    N_AUG_PER_IMAGE = 2  # 2 versions corrompues aleatoires par image d'origine

    for img, label in zip(train_crops, train_labels):
        for _ in range(N_AUG_PER_IMAGE):
            fn = CORRUPTION_FNS[rng.integers(0, len(CORRUPTION_FNS))]
            severity = rng.integers(1, 4)  # 1, 2 ou 3
            try:
                corrupted = fn(img, severity)
                corrupted_crops.append(corrupted)
                corrupted_labels.append(label)
            except Exception:
                continue

    print(f"Versions corrompues generees: {len(corrupted_crops)}")

    print("\nExtraction des embeddings sur les images corrompues (peut prendre quelques minutes)...")
    X_corrupted = extract_embeddings(dino, processor, corrupted_crops)
    y_corrupted = np.array(corrupted_labels)
    np.save("data/plantdoc_train_corrupted_embeddings.npy", X_corrupted)
    np.save("data/plantdoc_train_corrupted_labels.npy", y_corrupted)
    print("Embeddings corrompus sauvegardes.")

    # --- Reconstruction du dataset d'entrainement complet + versions corrompues ---
    print("\nPreparation du dataset d'entrainement robuste...")
    X_pd_full = np.load("data/plantdoc_embeddings.npy")
    y_pd_full = np.load("data/plantdoc_labels.npy")
    X_pd_train = X_pd_full[train_idx]
    y_pd_train = y_pd_full[train_idx]

    modelval_idx = np.load("data/production_modelval_idx.npy")
    X_pd_modelval = X_pd_full[modelval_idx]
    y_pd_modelval = y_pd_full[modelval_idx]

    X_pv_train = np.load("data/train_embeddings.npy")
    y_pv_train = np.load("data/train_labels.npy")
    X_val = np.load("data/val_embeddings.npy")
    y_val = np.load("data/val_labels.npy")

    X_combined = np.concatenate([X_pv_train, X_pd_train, X_pd_train, X_pd_train, X_corrupted], axis=0)
    y_combined = np.concatenate([y_pv_train, y_pd_train, y_pd_train, y_pd_train, y_corrupted], axis=0)
    print(f"Dataset d'entrainement robuste: {len(y_combined)} images (dont {len(y_corrupted)} corrompues)")

    from collections import Counter
    counts = Counter(y_combined.tolist())
    total = len(y_combined)
    class_weights = np.array([total / (n_classes * counts.get(i, 1)) for i in range(n_classes)], dtype=np.float32)
    weights_tensor = torch.tensor(class_weights)

    X_train_t = torch.tensor(X_combined, dtype=torch.float32)
    y_train_t = torch.tensor(y_combined, dtype=torch.long)
    X_pdval_t = torch.tensor(X_pd_modelval, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    torch.manual_seed(42)
    model = MLPHead(384, n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)

    n_epochs = 40
    batch_size = 256
    n_samples = X_train_t.shape[0]
    best_val_acc = 0.0

    print(f"\nEntrainement du modele ROBUSTE ({n_epochs} epochs)...")
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
            pdval_preds = torch.argmax(model(X_pdval_t), dim=1).numpy()
        pdval_acc = accuracy_score(y_pd_modelval, pdval_preds)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{n_epochs} | loss={avg_loss:.4f} | val_acc(PlantDoc)={pdval_acc:.4f}")

        if pdval_acc > best_val_acc:
            best_val_acc = pdval_acc
            torch.save(model.state_dict(), "models/PRODUCTION_MODEL_ROBUST.pt")

    print(f"\nModele robuste sauvegarde: models/PRODUCTION_MODEL_ROBUST.pt")
    print("\nProchaine etape: relancer robustness_test.py avec ce nouveau modele pour comparer.")

if __name__ == "__main__":
    main()
