"""
Script de prediction sur une photo unique - VERSION FINALE ROBUSTE.
Pipeline: DINOv2 embedding + Test-Time Augmentation + MLP robuste + severite par segmentation TTA + Multi-maladies.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import json
import cv2
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as TF
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

MODEL_NAME = "facebook/dinov2-small"

def get_leaf_mask_via_attention(pil_image, backbone, processor, target_size):
    inputs = processor(images=pil_image, return_tensors="pt").to(backbone.device)
    with torch.no_grad():
        outputs = backbone(**inputs, output_attentions=True)

    last_attention = outputs.attentions[-1][0]
    cls_attention = last_attention[:, 0, 1:].mean(dim=0).cpu()

    n_patches = cls_attention.shape[0]
    grid_size = int(n_patches ** 0.5)
    attn_map = cls_attention.reshape(grid_size, grid_size).numpy()

    # Calcul de la variance spatiale pour detecter si c'est un objet (feuille) ou du bruit/fond
    attn_variance = np.var(attn_map)
    is_diffuse = attn_variance < 1e-5  # Seuil empirique de rejet

    attn_resized = cv2.resize(attn_map, target_size, interpolation=cv2.INTER_CUBIC)
    attn_norm = ((attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8) * 255).astype(np.uint8)
    _, mask = cv2.threshold(attn_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return mask > 0, is_diffuse

def estimate_lesion_ratio(pil_image, backbone=None, processor=None, crop_fraction=0.65, predicted_class=""):
    img_full = np.array(pil_image.convert("RGB"))
    h, w = img_full.shape[:2]

    is_diffuse = False
    if backbone is not None and processor is not None:
        leaf_mask, is_diffuse = get_leaf_mask_via_attention(pil_image, backbone, processor, (w, h))
        img = img_full
    else:
        ch, cw = int(h * crop_fraction), int(w * crop_fraction)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        img = img_full[y0:y0+ch, x0:x0+cw]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        leaf_mask = leaf_mask > 0

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    specular_mask = (v_channel > 200) & (s_channel < 30)

    leaf_mask = leaf_mask & (~specular_mask)
    leaf_area = leaf_mask.sum()
    if leaf_area < 100:
        leaf_mask = np.ones(hsv.shape[:2], dtype=bool) & (~specular_mask)
        leaf_area = max(leaf_mask.sum(), 1)

    # Severite conditionnelle
    lower_healthy = np.array([25, 40, 40])
    upper_healthy = np.array([90, 255, 255])

    # Ajustement pour Myrtilles / Automne / classes avec teintes specifiques
    if "Blueberry" in predicted_class or "Cherry" in predicted_class or "Peach" in predicted_class:
        # Elargir la plage saine pour inclure des teintes marron/rouge (automne) ou violettes
        lower_healthy = np.array([10, 30, 30])
        upper_healthy = np.array([160, 255, 255])
    
    # Si le modele est tres confiant que c'est sain, on evite les faux positifs
    if "healthy" in predicted_class:
        # On tolere plus de variations
        lower_healthy[0] = max(0, lower_healthy[0] - 10)
        upper_healthy[0] = min(179, upper_healthy[0] + 10)

    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy) > 0
    diseased_mask = leaf_mask & (~healthy_mask)
    ratio = diseased_mask.sum() / max(leaf_area, 1)
    
    return float(np.clip(ratio, 0.0, 1.0)), is_diffuse

def calculate_severity_tta(pil_image, backbone, processor, predicted_class):
    views = get_tta_views(pil_image)
    ratios = []
    any_diffuse = False
    for view in views:
        ratio, is_diffuse = estimate_lesion_ratio(view, backbone, processor, predicted_class=predicted_class)
        ratios.append(ratio)
        if is_diffuse:
            any_diffuse = True
    return float(np.mean(ratios)), any_diffuse

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

def mahalanobis_min_distance(x, prototypes, cov_inv):
    diffs = prototypes - x
    dists_sq = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
    return float(np.sqrt(np.maximum(dists_sq, 0)).min())

def temperature_scaling(logits, temperature=1.5):
    return logits / temperature

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py chemin/vers/photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Chargement de l'image: {image_path}")
    
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erreur de chargement de l'image (fichier corrompu ou format non supporte): {e}")
        sys.exit(1)

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dino_model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
    dino_model.eval()

    embedding_dim = 384
    model_path = "models/PRODUCTION_MODEL_ROBUST.pt"
    clf = MLPHead(embedding_dim, n_classes)
    clf.load_state_dict(torch.load(model_path, map_location="cpu"))
    clf.eval()

    class_prototypes = np.load("models/class_prototypes.npy")
    mahalanobis_cov_inv = np.load("models/mahalanobis_cov_inv.npy")
    with open("models/ood_threshold.json", "r") as f:
        ood_threshold = json.load(f)["threshold"]

    views = get_tta_views(pil_image)
    inputs = processor(images=views, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        
        logits = clf(embeddings)
        # Temperature scaling pour mitiger l'overconfidence
        scaled_logits = temperature_scaling(logits, temperature=1.3)
        probs = torch.softmax(scaled_logits, dim=1)
        avg_probs = probs.mean(dim=0)

    # OOD Detection
    avg_embedding_np = embeddings.mean(dim=0).numpy()
    min_dist = mahalanobis_min_distance(avg_embedding_np, class_prototypes, mahalanobis_cov_inv)
    
    # Penalite de confiance si OOD
    if min_dist > ood_threshold:
        penalty = (min_dist - ood_threshold) / ood_threshold
        # On ecrase la probabilite max vers une distribution plus uniforme
        avg_probs = avg_probs * (1 - min(penalty, 0.5))

    avg_probs = avg_probs / avg_probs.sum() # re-normalize
    top5_probs, top5_idx = torch.topk(avg_probs, 5)
    
    predicted_class_name = labels_names[top5_idx[0].item()]

    # Multi-disease check
    diseases_found = []
    for prob, idx in zip(top5_probs, top5_idx):
        if prob.item() > 0.15 and "healthy" not in labels_names[idx]:
            diseases_found.append(labels_names[idx])
            
    is_multi_disease = len(diseases_found) > 1

    print("\n" + "="*60)
    print("=== DIAGNOSTIC ===")
    print("="*60)
    
    is_ood = min_dist > ood_threshold
    if is_ood:
        print(f"\n[⚠️] ALERTE OOD: L'image est eloignee des donnees d'entrainement (Dist: {min_dist:.1f} > Seuil: {ood_threshold:.1f}).")
        
    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_idx), 1):
        marker = " <-- PREDICTION" if rank == 1 else ""
        print(f"{rank}. {labels_names[idx]:<55} {prob.item()*100:5.1f}%{marker}")

    if is_multi_disease:
        print("\n[⚠️] ALERTE MULTI-MALADIES: Plusieurs maladies potentielles detectees simultanement :")
        for d in diseases_found:
            print(f"  - {d}")

    print("\n" + "="*60)
    print("=== SEVERITE (TTA + Conditionnelle) ===")
    print("="*60)
    
    ratio, is_diffuse = calculate_severity_tta(pil_image, dino_model, processor, predicted_class_name)
    sev_id, sev_name = ratio_to_severity(ratio)
    
    if is_diffuse:
        print("[⚠️] ALERTE NON-FEUILLE: La variance de l'attention est tres faible.")
        print("Il s'agit probablement d'un objet non pertinent (main, sol, ciel, regle).")
        print("La severite et la prediction peuvent etre faussees.")
        
    print(f"Ratio de surface atteinte estime: {ratio*100:.1f}%")
    print(f"Niveau de severite: {sev_name}")

if __name__ == "__main__":
    main()
