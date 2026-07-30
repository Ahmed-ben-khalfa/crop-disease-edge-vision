"""
Semaine 7 - Explicabilite: cartes d'attention DINOv2.
Visualise ou le modele "regarde" sur une image, pour verifier qu'il se concentre
sur les symptomes de la maladie et pas sur des artefacts (fond, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
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

def get_attention_map(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Attention de la derniere couche, moyennee sur toutes les tetes
    # Shape: (1, n_heads, n_tokens, n_tokens) -> on prend l'attention du CLS vers les patches
    last_attention = outputs.attentions[-1][0]  # (n_heads, n_tokens, n_tokens)
    cls_attention = last_attention[:, 0, 1:]  # attention du CLS vers tous les patches, toutes tetes
    cls_attention = cls_attention.mean(dim=0)  # moyenne sur les tetes

    n_patches = cls_attention.shape[0]
    grid_size = int(n_patches ** 0.5)
    attention_map = cls_attention.reshape(grid_size, grid_size).numpy()
    return attention_map

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Chargement de DINOv2 (avec sortie des attentions)...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    print("Selection de 6 exemples varies du test set...")
    test_idx = np.load("data/production_test_idx.npy")
    rng = np.random.default_rng(7)
    sample_pos = rng.choice(len(test_idx), size=6, replace=False)
    sample_group_idx = test_idx[sample_pos]
    crops, labels = rebuild_crops_sample(name_to_id, sample_group_idx)

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    for i, (img, label_id) in enumerate(zip(crops, labels)):
        print(f"Traitement image {i+1}/6: {labels_names[label_id]}")
        attention_map = get_attention_map(model, processor, img)

        axes[0, i].imshow(img)
        axes[0, i].set_title(labels_names[label_id], fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(img)
        axes[1, i].imshow(attention_map, cmap="jet", alpha=0.5, extent=(0, img.width, img.height, 0))
        axes[1, i].set_title("Carte d'attention", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("docs/attention_maps.png", dpi=150, bbox_inches="tight")
    print("\nCartes d'attention sauvegardees: docs/attention_maps.png")
    print("\nInterpretation attendue: les zones rouges/jaunes (forte attention)")
    print("devraient se concentrer sur les symptomes visibles de la maladie")
    print("(taches, decoloration) plutot que sur le fond ou les bords de l'image.")
    print("Ouvre le fichier PNG pour inspection visuelle.")

if __name__ == "__main__":
    main()
