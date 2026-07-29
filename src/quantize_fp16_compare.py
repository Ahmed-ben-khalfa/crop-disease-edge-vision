"""
Comparaison FP32 vs INT8 (deja teste, -5.0 points) vs FP16 (alternative plus douce).
Objectif: trouver le meilleur compromis taille/vitesse/precision qui passe
le seuil de securite (< 3 points de perte, > 90% d'accord avec FP32).
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset
from sklearn.metrics import accuracy_score

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

def rebuild_test_crops(name_to_id, test_group_idx):
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
    test_crops = [all_crops[i] for i in test_group_idx]
    test_labels = [all_labels[i] for i in test_group_idx]
    return test_crops, test_labels

def get_state_dict_size_mb(model):
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)

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

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Chargement du modele FP32 (PRODUCTION_MODEL)...")
    model_fp32 = UnifiedModel(n_classes)
    model_fp32.head.load_state_dict(torch.load("models/PRODUCTION_MODEL.pt", map_location="cpu"))
    model_fp32.eval()

    print("Conversion en FP16 (demi-precision)...")
    model_fp16 = UnifiedModel(n_classes)
    model_fp16.load_state_dict(model_fp32.state_dict())
    model_fp16 = model_fp16.half()
    model_fp16.eval()

    size_fp32 = get_state_dict_size_mb(model_fp32)
    size_fp16 = get_state_dict_size_mb(model_fp16)
    print(f"\n=== TAILLE ===")
    print(f"FP32: {size_fp32:.1f} Mo | FP16: {size_fp16:.1f} Mo | Reduction: {(1-size_fp16/size_fp32)*100:.1f}%")

    dummy_fp32 = torch.randn(1, 3, 224, 224)
    dummy_fp16 = dummy_fp32.half()
    lat_fp32 = measure_latency(model_fp32, dummy_fp32)
    lat_fp16 = measure_latency(model_fp16, dummy_fp16)
    print(f"\n=== LATENCE (FP16 sur CPU n'accelere pas toujours, contrairement au GPU) ===")
    print(f"FP32: {lat_fp32:.1f} ms | FP16: {lat_fp16:.1f} ms")

    print("\n" + "="*70)
    print("VALIDATION DE BOUT EN BOUT: FP16 vs FP32")
    print("="*70)
    test_group_idx = np.load("data/production_test_idx.npy")
    test_crops, test_labels = rebuild_test_crops(name_to_id, test_group_idx)

    SAMPLE_SIZE = 300
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(test_crops), size=min(SAMPLE_SIZE, len(test_crops)), replace=False)
    sample_crops = [test_crops[i] for i in sample_idx]
    sample_labels = np.array([test_labels[i] for i in sample_idx])

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    inputs = processor(images=sample_crops, return_tensors="pt")
    pixel_values_fp32 = inputs["pixel_values"]
    pixel_values_fp16 = pixel_values_fp32.half()

    print("Inference FP32...")
    preds_fp32 = []
    with torch.no_grad():
        for i in range(0, len(pixel_values_fp32), 16):
            logits = model_fp32(pixel_values_fp32[i:i+16])
            preds_fp32.extend(torch.argmax(logits, dim=1).tolist())

    print("Inference FP16...")
    preds_fp16 = []
    with torch.no_grad():
        for i in range(0, len(pixel_values_fp16), 16):
            logits = model_fp16(pixel_values_fp16[i:i+16])
            preds_fp16.extend(torch.argmax(logits.float(), dim=1).tolist())

    acc_fp32 = accuracy_score(sample_labels, preds_fp32)
    acc_fp16 = accuracy_score(sample_labels, preds_fp16)
    agreement = accuracy_score(preds_fp32, preds_fp16)

    print(f"\n*** RESULTAT FP16 (echantillon n={len(sample_crops)}) ***")
    print(f"Accuracy FP32: {acc_fp32:.4f} | Accuracy FP16: {acc_fp16:.4f}")
    print(f"Difference: {(acc_fp16-acc_fp32)*100:+.2f} points")
    print(f"Accord FP32/FP16: {agreement*100:.1f}%")

    print("\n" + "="*70)
    print("=== TABLEAU RECAPITULATIF DES OPTIONS ===")
    print("="*70)
    print(f"FP32 (base)        : {size_fp32:.1f} Mo | {lat_fp32:.1f} ms | reference")
    print(f"INT8 (teste avant) : 23.7 Mo | 34.2 ms | -5.0 points -> REJETE (seuil depasse)")
    verdict = "ACCEPTABLE" if abs(acc_fp16-acc_fp32) < 0.03 else "A EVALUER"
    print(f"FP16 (ce test)     : {size_fp16:.1f} Mo | {lat_fp16:.1f} ms | {(acc_fp16-acc_fp32)*100:+.2f} points -> {verdict}")

if __name__ == "__main__":
    main()
