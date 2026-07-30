"""
Semaine 7 (amelioration) - Explicabilite via Grad-CAM adapte aux Vision Transformers.
Plus rigoureux que l'attention brute: montre les zones qui influencent VRAIMENT
la prediction de la classe, via le gradient du score par rapport aux patches.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset

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

def rebuild_crops_sample(name_to_id, indices):
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
    sel_crops = [all_crops[i] for i in indices]
    sel_labels = [all_labels[i] for i in indices]
    return sel_crops, sel_labels

def compute_gradcam(backbone, head, processor, image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.enable_grad():
        outputs = backbone(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, n_tokens, dim)
        last_hidden.retain_grad()

        cls_token = last_hidden[:, 0, :]
        logits = head(cls_token)
        pred_class = torch.argmax(logits, dim=1).item()
        score = logits[0, pred_class]

        backbone.zero_grad()
        head.zero_grad()
        score.backward()

    gradients = last_hidden.grad[0, 1:, :]      # gradients sur les patches (hors CLS)
    activations = last_hidden[0, 1:, :].detach()  # activations sur les patches

    # Grad-CAM adapte: poids = moyenne du gradient par canal, puis somme ponderee par patch
    weights = gradients.mean(dim=0)               # (dim,)
    cam = (weights * activations).sum(dim=-1)      # (n_patches,)
    cam = torch.relu(cam)                          # comme le Grad-CAM original

    n_patches = cam.shape[0]
    grid_size = int(n_patches ** 0.5)
    cam_map = cam.reshape(grid_size, grid_size).numpy()

    # Normalisation 0-1 pour la visualisation
    if cam_map.max() > cam_map.min():
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())

    return cam_map, pred_class

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}
    n_classes = split_info["n_classes"]

    print("Chargement du modele...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    head = MLPHead(384, n_classes)
    head.load_state_dict(torch.load("models/PRODUCTION_MODEL.pt", map_location="cpu"))
    backbone.eval()
    head.eval()

    print("Selection des 6 memes exemples que precedemment (seed=7)...")
    test_idx = np.load("data/production_test_idx.npy")
    rng = np.random.default_rng(7)
    sample_pos = rng.choice(len(test_idx), size=6, replace=False)
    sample_group_idx = test_idx[sample_pos]
    crops, labels = rebuild_crops_sample(name_to_id, sample_group_idx)

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    for i, (img, true_label_id) in enumerate(zip(crops, labels)):
        print(f"Traitement image {i+1}/6: vraie classe = {labels_names[true_label_id]}")
        cam_map, pred_class = compute_gradcam(backbone, head, processor, img)
        correct = "OK" if pred_class == true_label_id else "ERREUR"
        print(f"  Prediction: {labels_names[pred_class]} [{correct}]")

        axes[0, i].imshow(img)
        title = f"Vrai: {labels_names[true_label_id][:25]}\nPred: {labels_names[pred_class][:25]} [{correct}]"
        axes[0, i].set_title(title, fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(img)
        axes[1, i].imshow(cam_map, cmap="jet", alpha=0.55, extent=(0, img.width, img.height, 0))
        axes[1, i].set_title("Grad-CAM (base sur la prediction)", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("docs/gradcam_maps.png", dpi=150, bbox_inches="tight")
    print("\nGrad-CAM sauvegarde: docs/gradcam_maps.png")
    print("Cette fois, les zones chaudes indiquent ce qui a VRAIMENT motive la prediction du modele,")
    print("pas juste ou l'attention generique du backbone se porte.")

if __name__ == "__main__":
    main()
