import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
import cv2
from io import BytesIO
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as TF
import warnings
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# -----------------
# MODEL DEFS
# -----------------
MODEL_NAME = "facebook/dinov2-small"
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

# -----------------
# UTILS
# -----------------
def get_tta_views(pil_image):
    views = [pil_image, TF.hflip(pil_image), TF.vflip(pil_image)]
    w, h = pil_image.size
    crop_size = int(min(w, h) * 0.9)
    views.append(TF.center_crop(pil_image, crop_size))
    return views

def mahalanobis_min_distance(x, prototypes, cov_inv):
    diffs = prototypes - x
    dists_sq = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
    return float(np.sqrt(np.maximum(dists_sq, 0)).min())

def temperature_scaling(logits, temperature=1.3):
    return logits / temperature

def get_leaf_mask_via_attention(pil_image, backbone, processor, target_size):
    inputs = processor(images=pil_image, return_tensors="pt").to(backbone.device)
    with torch.no_grad():
        outputs = backbone(**inputs, output_attentions=True)

    last_attention = outputs.attentions[-1][0]
    cls_attention = last_attention[:, 0, 1:].mean(dim=0).cpu()

    n_patches = cls_attention.shape[0]
    grid_size = int(n_patches ** 0.5)
    attn_map = cls_attention.reshape(grid_size, grid_size).numpy()

    attn_variance = np.var(attn_map)
    is_diffuse = attn_variance < 1e-5

    attn_resized = cv2.resize(attn_map, target_size, interpolation=cv2.INTER_CUBIC)
    attn_norm = ((attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8) * 255).astype(np.uint8)
    _, mask = cv2.threshold(attn_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return mask > 0, is_diffuse

def estimate_lesion_ratio(pil_image, backbone, processor, predicted_class):
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]
    
    leaf_mask, is_diffuse = get_leaf_mask_via_attention(pil_image, backbone, processor, (w, h))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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

    if "Blueberry" in predicted_class or "Cherry" in predicted_class or "Peach" in predicted_class:
        lower_healthy = np.array([10, 30, 30])
        upper_healthy = np.array([160, 255, 255])
    
    if "healthy" in predicted_class:
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
        return 0, "Sain / très léger"
    elif ratio < 0.25:
        return 1, "Léger à modéré"
    else:
        return 2, "Sévère"

# -----------------
# GLOBAL STATE
# -----------------
app = FastAPI(title="Crop Disease API")

# Mount static files
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Globals
model_env = {}

@app.on_event("startup")
def load_models():
    print("Loading models and assets...")
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    model_env["labels_names"] = split_info["labels_names"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_env["device"] = device
    
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dino_model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
    dino_model.eval()
    
    clf = MLPHead(384, split_info["n_classes"])
    clf.load_state_dict(torch.load("models/PRODUCTION_MODEL_ROBUST.pt", map_location="cpu"))
    clf.eval()
    
    model_env["processor"] = processor
    model_env["dino_model"] = dino_model
    model_env["clf"] = clf
    model_env["class_prototypes"] = np.load("models/class_prototypes.npy")
    model_env["mahalanobis_cov_inv"] = np.load("models/mahalanobis_cov_inv.npy")
    
    with open("models/ood_threshold.json", "r") as f:
        model_env["ood_threshold"] = json.load(f)["threshold"]
        
    print("Models loaded successfully.")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier invalide, veuillez uploader une image.")
    
    try:
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image illisible ou corrompue: {str(e)}")

    device = model_env["device"]
    processor = model_env["processor"]
    dino_model = model_env["dino_model"]
    clf = model_env["clf"]
    labels_names = model_env["labels_names"]
    class_prototypes = model_env["class_prototypes"]
    cov_inv = model_env["mahalanobis_cov_inv"]
    ood_threshold = model_env["ood_threshold"]
    
    # 1. Prediction (TTA)
    views = get_tta_views(pil_image)
    inputs = processor(images=views, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        
        logits = clf(embeddings)
        scaled_logits = temperature_scaling(logits, temperature=1.3)
        probs = torch.softmax(scaled_logits, dim=1)
        avg_probs = probs.mean(dim=0)

    # 2. OOD Detection & Penalization
    avg_embedding_np = embeddings.mean(dim=0).numpy()
    min_dist = mahalanobis_min_distance(avg_embedding_np, class_prototypes, cov_inv)
    is_ood = min_dist > ood_threshold
    
    if is_ood:
        penalty = (min_dist - ood_threshold) / ood_threshold
        avg_probs = avg_probs * (1 - min(penalty, 0.5))
        avg_probs = avg_probs / avg_probs.sum()

    top5_probs, top5_idx = torch.topk(avg_probs, 5)
    predictions = [{"label": labels_names[idx.item()], "probability": float(prob.item())} for prob, idx in zip(top5_probs, top5_idx)]
    
    top1_class = predictions[0]["label"]

    # 3. Multi-disease
    diseases_found = []
    for p in predictions:
        if p["probability"] > 0.15 and "healthy" not in p["label"]:
            diseases_found.append(p["label"])
            
    is_multi_disease = len(diseases_found) > 1

    # 4. Severity (TTA + Conditional)
    ratio, is_diffuse = calculate_severity_tta(pil_image, dino_model, processor, top1_class)
    sev_id, sev_name = ratio_to_severity(ratio)

    return {
        "predictions": predictions,
        "is_ood": is_ood,
        "ood_distance": min_dist,
        "ood_threshold": ood_threshold,
        "is_multi_disease": is_multi_disease,
        "multi_diseases": diseases_found,
        "is_non_leaf": is_diffuse,
        "severity": {
            "ratio": ratio,
            "level": sev_id,
            "description": sev_name
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
