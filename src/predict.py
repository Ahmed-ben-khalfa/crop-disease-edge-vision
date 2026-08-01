"""
Script de prediction sur une photo unique - VERSION FINALE ROBUSTE.
Pipeline: DINOv2 embedding + Test-Time Augmentation + MLP robuste (corruption-augmented) + severite par segmentation.

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

def get_leaf_mask_via_attention(pil_image, backbone, processor, target_size):
    """
    IDEE: utiliser l'attention native de DINOv2 (propriete emergente documentee,
    Caron et al. 2021, "Emerging Properties in Self-Supervised Vision Transformers")
    comme segmenteur du sujet principal (la feuille), au lieu d'un seuillage couleur
    naif qui confond terre/copeaux/reflets d'eau avec la feuille.
    """
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = backbone(**inputs, output_attentions=True)

    last_attention = outputs.attentions[-1][0]
    cls_attention = last_attention[:, 0, 1:].mean(dim=0)

    n_patches = cls_attention.shape[0]
    grid_size = int(n_patches ** 0.5)
    attn_map = cls_attention.reshape(grid_size, grid_size).numpy()

    attn_resized = cv2.resize(attn_map, target_size, interpolation=cv2.INTER_CUBIC)
    attn_norm = ((attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8) * 255).astype(np.uint8)
    _, mask = cv2.threshold(attn_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0

def estimate_lesion_ratio(pil_image, backbone=None, processor=None, crop_fraction=0.65):
    img_full = np.array(pil_image.convert("RGB"))
    h, w = img_full.shape[:2]

    if backbone is not None and processor is not None:
        leaf_mask = get_leaf_mask_via_attention(pil_image, backbone, processor, (w, h))
        img = img_full
    else:
        ch, cw = int(h * crop_fraction), int(w * crop_fraction)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        img = img_full[y0:y0+ch, x0:x0+cw]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        leaf_mask = leaf_mask > 0

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Exclusion des reflets/gouttes d'eau (haute luminosite, faible saturation)
    # Ce ne sont ni des zones saines ni des zones malades: on les retire du calcul
    # plutot que de les compter a tort comme "maladie" (comme observe en test aveugle).
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    specular_mask = (v_channel > 200) & (s_channel < 30)

    leaf_mask = leaf_mask & (~specular_mask)
    leaf_area = leaf_mask.sum()
    if leaf_area < 100:
        leaf_mask = np.ones(hsv.shape[:2], dtype=bool) & (~specular_mask)
        leaf_area = max(leaf_mask.sum(), 1)

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
    dino_model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
    dino_model.eval()

    embedding_dim = 384
    model_path = "models/PRODUCTION_MODEL_ROBUST.pt"
    clf = MLPHead(embedding_dim, n_classes)
    clf.load_state_dict(torch.load(model_path, map_location="cpu"))
    clf.eval()
    print(f"Modele charge: {model_path} (MLP robuste, augmentation de corruptions)")

    class_prototypes = np.load("models/class_prototypes.npy")
    mahalanobis_cov_inv = np.load("models/mahalanobis_cov_inv.npy")
    with open("models/ood_threshold.json", "r") as f:
        ood_threshold = json.load(f)["threshold"]

    def mahalanobis_min_distance(x, prototypes, cov_inv):
        diffs = prototypes - x
        dists_sq = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
        return float(np.sqrt(np.maximum(dists_sq, 0)).min())

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

    # --- Detection hors-distribution: distance de Mahalanobis (Lee et al. 2018) ---
    avg_embedding_np = embeddings.mean(dim=0).numpy()
    min_dist = mahalanobis_min_distance(avg_embedding_np, class_prototypes, mahalanobis_cov_inv)
    is_ood = min_dist > ood_threshold

    print("\n" + "="*60)
    print("=== DIAGNOSTIC (moyenne sur", len(views), "vues TTA) ===")
    print("="*60)
    if is_ood:
        print(f"\n*** ATTENTION: photo hors-distribution probable ***")
        print(f"Distance au prototype le plus proche: {min_dist:.1f} (seuil: {ood_threshold:.1f})")
        print("Cette photo est visuellement tres differente des donnees d'entrainement.")
        print("Il est possible qu'il s'agisse d'une culture non couverte (14 cultures connues).")
        print("Le diagnostic ci-dessous est PEU FIABLE.\n")
    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_idx), 1):
        marker = " <-- PREDICTION" if rank == 1 else ""
        print(f"{rank}. {labels_names[idx]:<55} {prob.item()*100:5.1f}%{marker}")

    print("\n" + "="*60)
    print("=== SEVERITE (segmentation couleur, independante du modele) ===")
    print("="*60)
    ratio = estimate_lesion_ratio(pil_image, backbone=dino_model, processor=processor)
    sev_id, sev_name = ratio_to_severity(ratio)
    print(f"Ratio de surface atteinte estime: {ratio*100:.1f}%")
    print(f"Niveau de severite: {sev_name} (niveau {sev_id}/2)")

    print("\n" + "="*60)
    print("NOTE: prediction basee sur DINOv2-small + MLP robuste (PlantVillage+PlantDoc")
    print("+ augmentation par corruptions flou/bruit/JPEG).")
    print("Performance mesuree sur echantillon de validation (200 images, sans fuite):")
    print("  PlantDoc (terrain, images propres): 72.5% accuracy")
    print("  PlantVillage (labo): ~97-98% accuracy")
    print("  Robustesse: +3 a +15 points vs modele original selon le type de corruption")
    print("La confiance affichee est un indicateur, pas une garantie absolue.")
    print("="*60)

if __name__ == "__main__":
    main()
