"""
Extraction complete des embeddings DINOv2 sur train/val/test.
Semaine 2 - Etape 1 (suite) : Extraction des features sur tout le dataset.
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
import numpy as np
import json
import time

MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 32

def extract_embeddings(model, processor, images, device, labels_ref=""):
    embeddings = []
    n = len(images)
    start = time.time()

    for i in range(0, n, BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

        if i % (BATCH_SIZE * 20) == 0 and i > 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (n - i) / rate if rate > 0 else 0
            print(f"  [{labels_ref}] {i}/{n} images traitees ({rate:.1f} img/s, ETA: {eta/60:.1f} min)")

    return np.concatenate(embeddings, axis=0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilise: {device}")

    print(f"\nChargement de {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("\nChargement du dataset et des indices de split...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    train_full = dataset["train"]
    test_full = dataset["test"]

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)

    train_idx = split_info["train_indices"]
    val_idx = split_info["val_indices"]

    # --- TRAIN ---
    print(f"\n=== Extraction TRAIN ({len(train_idx)} images) ===")
    train_images = [train_full[i]["image"].convert("RGB") for i in train_idx]
    train_labels = [train_full[i]["label"] for i in train_idx]
    train_embeddings = extract_embeddings(model, processor, train_images, device, "TRAIN")
    np.save("data/train_embeddings.npy", train_embeddings)
    np.save("data/train_labels.npy", np.array(train_labels))
    print(f"Sauvegarde: data/train_embeddings.npy {train_embeddings.shape}")

    # --- VAL ---
    print(f"\n=== Extraction VAL ({len(val_idx)} images) ===")
    val_images = [train_full[i]["image"].convert("RGB") for i in val_idx]
    val_labels = [train_full[i]["label"] for i in val_idx]
    val_embeddings = extract_embeddings(model, processor, val_images, device, "VAL")
    np.save("data/val_embeddings.npy", val_embeddings)
    np.save("data/val_labels.npy", np.array(val_labels))
    print(f"Sauvegarde: data/val_embeddings.npy {val_embeddings.shape}")

    # --- TEST ---
    print(f"\n=== Extraction TEST ({len(test_full)} images) ===")
    test_images = [img.convert("RGB") for img in test_full["image"]]
    test_labels = test_full["label"]
    test_embeddings = extract_embeddings(model, processor, test_images, device, "TEST")
    np.save("data/test_embeddings.npy", test_embeddings)
    np.save("data/test_labels.npy", np.array(test_labels))
    print(f"Sauvegarde: data/test_embeddings.npy {test_embeddings.shape}")

    print("\n=== EXTRACTION TERMINEE ===")
    print("Tous les embeddings sont sauvegardes dans data/")

if __name__ == "__main__":
    main()
