"""
Script de prediction sur une photo unique - VERSION FINALE (modele propre, sans fuite).
Pipeline: DINOv2 embedding + Test-Time Augmentation + MLP entraine (final_corrected_best.pt) + severite par segmentation.

Usage: python src/predict.py chemin/vers/photo.jpg
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import json
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as TF

MODEL_NAME = "facebook/dinov2-small"

def estimate_lesion_ratio(pil_image):
    img = np.array(pil_image.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    leaf_mask = leaf_mask > 0
    leaf_area = leaf_mask.sum()
    if leaf_area < 100:
        leaf_mask = np.ones(gray.shape, dtype=bool)
        leaf_area = leaf_mask.sum()
    lower_healthy = np.array([25, 40, 40])
    upper_healthy = np.array([90, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy) > 0
    diseased_mask = leaf_mask & (~healthy_mask)
    ratio = diseased_mask.sum() / max(leaf_area, 1)
    return float(np.clip(ratio, 0.0, 1.0))

def ratio_to_severity(ratio):
    if ratio < 0.05:
        return 0, "Sain / tres leger"
    elif ratio < 0.25:
        return 1, "Leger a modere"
    else:
        return 2, "Severe"

def get_tta_views(pil_image):
    views = [pil_image]
    views.append(TF.hflip(pil_image))
    views.append(TF.vflip(pil_image))
    w, h = pil_image.size
    crop_size = int(min(w, h) * 0.9)
    views.append(TF.center_crop(pil_image, crop_size))
    return views

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
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py chemin/vers/photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Chargement de l'image: {image_path}")
    pil_image = Image.open(image_path).convert("RGB")

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de {MODEL_NAME} sur {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dino_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    dino_model.eval()

    embedding_dim = 384
    model_path = "models/final_corrected_best.pt"
    clf = MLPHead(embedding_dim, n_classes)
    clf.load_state_dict(torch.load(model_path, map_location="cpu"))
    clf.eval()
    print(f"Modele charge: {model_path} (MLP, PlantVillage+PlantDoc, sans fuite - 69.2% +/- 0.6% sur PlantDoc)")

    print("\nGeneration des vues TTA (image originale + flips + crop centre)...")
    views = get_tta_views(pil_image)
    print(f"{len(views)} vues generees.")

    print("Extraction des embeddings DINOv2 sur chaque vue...")
    inputs = processor(images=views, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()

    with torch.no_grad():
        logits = clf(embeddings)
        probs = torch.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0)

    top5_probs, top5_idx = torch.topk(avg_probs, 5)

    print("\n" + "="*60)
    print("=== DIAGNOSTIC (moyenne sur", len(views), "vues TTA) ===")
    print("="*60)
    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_idx), 1):
        marker = " <-- PREDICTION" if rank == 1 else ""
        print(f"{rank}. {labels_names[idx]:<55} {prob.item()*100:5.1f}%{marker}")

    print("\n" + "="*60)
    print("=== SEVERITE (segmentation couleur, independante du modele) ===")
    print("="*60)
    ratio = estimate_lesion_ratio(pil_image)
    sev_id, sev_name = ratio_to_severity(ratio)
    print(f"Ratio de surface atteinte estime: {ratio*100:.1f}%")
    print(f"Niveau de severite: {sev_name} (niveau {sev_id}/2)")

    print("\n" + "="*60)
    print("NOTE: prediction basee sur DINOv2-small + MLP entraine sur PlantVillage+PlantDoc.")
    print("Performance mesuree (validation croisee, 3 seeds, sans fuite de donnees):")
    print("  PlantDoc (terrain): 69.2% +/- 0.6% accuracy")
    print("  PlantVillage (labo): ~97-98% accuracy")
    print("La confiance affichee est un indicateur, pas une garantie absolue.")
    print("="*60)

if __name__ == "__main__":
    main()
