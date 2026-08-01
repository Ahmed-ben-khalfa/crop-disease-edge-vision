"""
App de demo interactive - Diagnostic de maladies des plantes.
Interface web locale (glisser-deposer une photo, voir le diagnostic instantanement).
Simule l'experience "app mobile" pour une demo/video de portfolio.

Installation: pip install gradio
Lancement: python src/demo_app.py
Puis ouvrir le lien affiche (http://127.0.0.1:7860) dans un navigateur.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as TF
import gradio as gr

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
        return "🟢 Sain / tres leger"
    elif ratio < 0.25:
        return "🟡 Leger a modere"
    else:
        return "🔴 Severe"

def get_tta_views(pil_image):
    views = [pil_image, TF.hflip(pil_image), TF.vflip(pil_image)]
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

print("Chargement du modele (une seule fois, au demarrage de l'app)...")
with open("data/split_info.json", "r") as f:
    split_info = json.load(f)
LABELS_NAMES = split_info["labels_names"]
N_CLASSES = split_info["n_classes"]

PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)
DINO_MODEL = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
DINO_MODEL.eval()

CLF = MLPHead(384, N_CLASSES)
CLF.load_state_dict(torch.load("models/PRODUCTION_MODEL_ROBUST.pt", map_location="cpu"))
CLF.eval()

CLASS_PROTOTYPES = np.load("models/class_prototypes.npy")
MAHALANOBIS_COV_INV = np.load("models/mahalanobis_cov_inv.npy")
with open("models/ood_threshold.json", "r") as f:
    OOD_THRESHOLD = json.load(f)["threshold"]
print(f"Prototypes et seuil OOD (Mahalanobis) charges (seuil={OOD_THRESHOLD:.2f}).")
print("Modele pret.")

def mahalanobis_min_distance(x, prototypes, cov_inv):
    diffs = prototypes - x
    dists_sq = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
    return float(np.sqrt(np.maximum(dists_sq, 0)).min())

def predict(pil_image):
    if pil_image is None:
        return "Merci d'uploader une photo.", "", None

    pil_image = pil_image.convert("RGB")
    views = get_tta_views(pil_image)
    inputs = PROCESSOR(images=views, return_tensors="pt")
    with torch.no_grad():
        outputs = DINO_MODEL(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        logits = CLF(embeddings)
        probs = torch.softmax(logits, dim=1).mean(dim=0)

    top5_probs, top5_idx = torch.topk(probs, 5)
    results = {LABELS_NAMES[idx]: prob.item() for prob, idx in zip(top5_probs, top5_idx)}

    ratio = estimate_lesion_ratio(pil_image, backbone=DINO_MODEL, processor=PROCESSOR)
    severity_text = ratio_to_severity(ratio)
    severity_full = f"{severity_text}\n\nSurface foliaire atteinte estimee : {ratio*100:.1f}%"

    # --- Detection hors-distribution: distance de Mahalanobis (Lee et al. 2018) ---
    avg_embedding_np = embeddings.mean(dim=0).numpy()
    min_dist = mahalanobis_min_distance(avg_embedding_np, CLASS_PROTOTYPES, MAHALANOBIS_COV_INV)

    # Affichage systematique de la distance (diagnostic), meme si sous le seuil
    severity_full += f"\n\n[Debug OOD] distance Mahalanobis = {min_dist:.2f} (seuil = {OOD_THRESHOLD:.2f})"

    if min_dist > OOD_THRESHOLD:
        warning = (
            f"\n\n⚠️ ATTENTION: cette photo est visuellement tres differente de nos "
            f"donnees d'entrainement (distance={min_dist:.1f}, seuil={OOD_THRESHOLD:.1f}).\n"
            f"Il est possible qu'il s'agisse d'une culture non couverte par ce modele "
            f"(seules 14 cultures sont connues). Le diagnostic ci-dessus est peu fiable."
        )
        severity_full += warning

    return results, severity_full

with gr.Blocks(title="Diagnostic Maladies des Plantes") as demo:
    gr.Markdown("# 🌾 Diagnostic de Maladies des Plantes par IA")
    gr.Markdown(
        "Prends une photo d'une feuille (ou uploade une image existante) pour obtenir un "
        "diagnostic instantane. Modele DINOv2 + tete de classification, entraine sur "
        "PlantVillage + PlantDoc (72.5% accuracy en conditions terrain, mesure honnetement)."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Photo de la feuille")
            submit_btn = gr.Button("Diagnostiquer", variant="primary")
        with gr.Column():
            output_labels = gr.Label(label="Diagnostic (top 5)", num_top_classes=5)
            output_severity = gr.Textbox(label="Severite estimee", lines=3)

    submit_btn.click(fn=predict, inputs=image_input, outputs=[output_labels, output_severity])
    image_input.change(fn=predict, inputs=image_input, outputs=[output_labels, output_severity])

    gr.Markdown(
        "---\n"
        "*Note: prediction basee sur un modele de recherche, pas un dispositif medical/agronomique certifie. "
        "En cas de doute, consulter un expert agronome.*"
    )

if __name__ == "__main__":
    demo.launch()
