"""
Test cible: quantifier UNIQUEMENT le backbone DINOv2 en INT8,
garder la tete MLP en FP32 (elle est minuscule, ~0.5 Mo, pas d'interet a la quantifier).
Hypothese: la degradation vient surtout du backbone, pas de la tete.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import copy

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

    print("Quantization SELECTIVE: backbone en INT8, tete gardee en FP32...")
    # On quantifie uniquement le sous-module backbone, pas le modele entier
    backbone_int8 = torch.quantization.quantize_dynamic(
        model_fp32.backbone, {nn.Linear}, dtype=torch.qint8
    )

    class HybridModel(nn.Module):
        def __init__(self, backbone_q, head_fp32):
            super().__init__()
            self.backbone = backbone_q
            self.head = head_fp32
        def forward(self, pixel_values):
            outputs = self.backbone(pixel_values=pixel_values)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return self.head(cls_embedding)

    model_hybrid = HybridModel(backbone_int8, model_fp32.head)
    model_hybrid.eval()

    size_fp32 = get_state_dict_size_mb(model_fp32)
    size_hybrid = get_state_dict_size_mb(model_hybrid)
    print(f"\n=== TAILLE ===")
    print(f"FP32 complet: {size_fp32:.1f} Mo")
    print(f"Hybride (backbone INT8 + tete FP32): {size_hybrid:.1f} Mo (reduction {(1-size_hybrid/size_fp32)*100:.1f}%)")

    print("\n" + "="*70)
    print("VALIDATION DE BOUT EN BOUT: Hybride vs FP32")
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
    pixel_values = inputs["pixel_values"]

    print("Inference FP32...")
    preds_fp32 = []
    with torch.no_grad():
        for i in range(0, len(pixel_values), 16):
            logits = model_fp32(pixel_values[i:i+16])
            preds_fp32.extend(torch.argmax(logits, dim=1).tolist())

    print("Inference Hybride (backbone INT8 + tete FP32)...")
    preds_hybrid = []
    with torch.no_grad():
        for i in range(0, len(pixel_values), 16):
            logits = model_hybrid(pixel_values[i:i+16])
            preds_hybrid.extend(torch.argmax(logits, dim=1).tolist())

    acc_fp32 = accuracy_score(sample_labels, preds_fp32)
    acc_hybrid = accuracy_score(sample_labels, preds_hybrid)
    agreement = accuracy_score(preds_fp32, preds_hybrid)

    print(f"\n*** RESULTAT HYBRIDE (echantillon n={len(sample_crops)}) ***")
    print(f"Accuracy FP32: {acc_fp32:.4f} | Accuracy Hybride: {acc_hybrid:.4f}")
    print(f"Difference: {(acc_hybrid-acc_fp32)*100:+.2f} points")
    print(f"Accord FP32/Hybride: {agreement*100:.1f}%")

    if abs(acc_hybrid - acc_fp32) < 0.03 and agreement > 0.90:
        print("\n*** VERDICT: ACCEPTABLE - ce modele hybride passe le seuil de securite ***")
        torch.save(model_hybrid.head.state_dict(), "models/hybrid_head_fp32.pt")
        torch.save(backbone_int8.state_dict(), "models/hybrid_backbone_int8.pt")
        print("Modele hybride sauvegarde: models/hybrid_backbone_int8.pt + hybrid_head_fp32.pt")
    else:
        print("\n*** VERDICT: TOUJOURS INSUFFISANT - la degradation vient bien du backbone lui-meme ***")
        print("Prochaine option a explorer: quantization statique calibree, ou QAT")

if __name__ == "__main__":
    main()
