"""
Semaine 5 - Etape 2 (CORRIGE) : Quantization INT8 avec validation de bout en bout.
Correction: on teste l'impact reel sur le BACKBONE (99% du modele),
pas seulement sur la petite tete MLP.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

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

class UnifiedModel(nn.Module):
    def __init__(self, n_classes, embedding_dim=384):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.head = MLPHead(embedding_dim, n_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)

def get_state_dict_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)

def measure_latency(model, dummy_input, n_runs=15):
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(dummy_input)
            times.append((time.time() - start) * 1000)
    return sum(times) / len(times)

def rebuild_test_crops(name_to_id, test_group_idx):
    """Reconstruit les vraies images (crops) correspondant aux indices du test set sans fuite."""
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
    test_crops = [all_crops[i] for i in test_group_idx]
    test_labels = [all_labels[i] for i in test_group_idx]
    return test_crops, test_labels

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Chargement du modele unifie FP32...")
    model_fp32 = UnifiedModel(n_classes)
    model_fp32.load_state_dict(torch.load("models/unified_model_fp32.pt", map_location="cpu"))
    model_fp32.eval()

    print("Application de la quantization dynamique INT8 (backbone + tete)...")
    model_int8 = torch.quantization.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)
    model_int8.eval()
    torch.save(model_int8.state_dict(), "models/unified_model_int8.pt")

    size_fp32 = get_state_dict_size_mb("models/unified_model_fp32.pt")
    size_int8 = get_state_dict_size_mb("models/unified_model_int8.pt")
    print(f"\n=== TAILLE ===")
    print(f"FP32: {size_fp32:.1f} Mo | INT8: {size_int8:.1f} Mo | Reduction: {(1-size_int8/size_fp32)*100:.1f}%")

    dummy_input = torch.randn(1, 3, 224, 224)
    lat_fp32 = measure_latency(model_fp32, dummy_input)
    lat_int8 = measure_latency(model_int8, dummy_input)
    print(f"\n=== LATENCE ===")
    print(f"FP32: {lat_fp32:.1f} ms | INT8: {lat_int8:.1f} ms | Facteur: {lat_fp32/lat_int8:.2f}x")

    # --- VALIDATION CORRIGEE: pipeline COMPLET (image brute -> prediction) sur un vrai echantillon ---
    print("\n" + "="*70)
    print("VALIDATION CORRIGEE: impact reel sur le BACKBONE (pas juste la tete)")
    print("="*70)

    if not os.path.exists("data/production_test_idx.npy"):
        print("ERREUR: data/production_test_idx.npy introuvable.")
        print("Lance d'abord src/fix_group_leakage.py")
        return

    test_group_idx = np.load("data/production_test_idx.npy")
    print(f"Reconstruction des images du test set ({len(test_group_idx)} crops)...")
    test_crops, test_labels = rebuild_test_crops(name_to_id, test_group_idx)

    # Echantillon pour rester rapide (validation, pas benchmark exhaustif)
    SAMPLE_SIZE = 300
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(test_crops), size=min(SAMPLE_SIZE, len(test_crops)), replace=False)
    sample_crops = [test_crops[i] for i in sample_idx]
    sample_labels = np.array([test_labels[i] for i in sample_idx])

    print(f"Echantillon de validation: {len(sample_crops)} images")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    inputs = processor(images=sample_crops, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    print("Inference FP32 (bout en bout, image brute -> prediction)...")
    preds_fp32 = []
    with torch.no_grad():
        for i in range(0, len(pixel_values), 16):
            batch = pixel_values[i:i+16]
            logits = model_fp32(batch)
            preds_fp32.extend(torch.argmax(logits, dim=1).tolist())

    print("Inference INT8 (bout en bout, image brute -> prediction)...")
    preds_int8 = []
    with torch.no_grad():
        for i in range(0, len(pixel_values), 16):
            batch = pixel_values[i:i+16]
            logits = model_int8(batch)
            preds_int8.extend(torch.argmax(logits, dim=1).tolist())

    acc_fp32 = accuracy_score(sample_labels, preds_fp32)
    acc_int8 = accuracy_score(sample_labels, preds_int8)
    agreement = accuracy_score(preds_fp32, preds_int8)  # a quel point les 2 modeles sont d'accord entre eux

    print(f"\n*** RESULTAT VALIDATION DE BOUT EN BOUT (echantillon n={len(sample_crops)}) ***")
    print(f"Accuracy FP32 (pipeline complet): {acc_fp32:.4f}")
    print(f"Accuracy INT8 (pipeline complet): {acc_int8:.4f}")
    print(f"Difference: {(acc_int8-acc_fp32)*100:+.2f} points")
    print(f"Taux d'accord FP32 vs INT8 (memes predictions): {agreement*100:.1f}%")
    print("\n(Si la difference est faible (<3 points) et l'accord eleve (>90%),")
    print("la quantization est consideree comme sure pour le deploiement.)")

if __name__ == "__main__":
    main()
