"""
Mapping PlantDoc -> PlantVillage, decoupage des bounding boxes,
extraction d'embeddings DINOv2, et evaluation du domain gap labo->terrain.
Semaine 3 - Etape 1 (fin) : Mesure du domain gap.
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn

MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 32

# Mapping categorie PlantDoc -> nom de classe PlantVillage (doit correspondre exactement)
CATEGORY_MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

def crop_boxes(dataset_split, labels_names, name_to_id):
    crops, labels = [], []
    skipped = 0
    for example in dataset_split:
        img = example["image"].convert("RGB")
        objects = example["objects"]
        for bbox, category in zip(objects["bbox"], objects["category"]):
            if category not in CATEGORY_MAPPING:
                skipped += 1
                continue
            mapped_name = CATEGORY_MAPPING[category]
            if mapped_name not in name_to_id:
                skipped += 1
                continue
            x, y, w, h = bbox
            crop = img.crop((x, y, x + w, y + h))
            crops.append(crop)
            labels.append(name_to_id[mapped_name])
    return crops, labels, skipped

def extract_embeddings(model, processor, images, device):
    embeddings = []
    n = len(images)
    for i in range(0, n, BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
        if i % (BATCH_SIZE * 10) == 0 and i > 0:
            print(f"  {i}/{n} images traitees...")
    return np.concatenate(embeddings, axis=0)

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Chargement de PlantDoc...")
    plantdoc = load_dataset("agyaatcoder/PlantDoc")

    print("\nDecoupage des bounding boxes (train + test PlantDoc combines)...")
    crops_train, labels_train, skip_train = crop_boxes(plantdoc["train"], labels_names, name_to_id)
    crops_test, labels_test, skip_test = crop_boxes(plantdoc["test"], labels_names, name_to_id)

    all_crops = crops_train + crops_test
    all_labels = labels_train + labels_test

    print(f"\nImages exploitables (mappees): {len(all_crops)}")
    print(f"Images ignorees (categorie sans equivalent): {skip_train + skip_test}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nChargement de {MODEL_NAME} sur {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print(f"\nExtraction des embeddings sur {len(all_crops)} crops PlantDoc...")
    embeddings = extract_embeddings(model, processor, all_crops, device)

    np.save("data/plantdoc_embeddings.npy", embeddings)
    np.save("data/plantdoc_labels.npy", np.array(all_labels))
    print("Embeddings sauvegardes: data/plantdoc_embeddings.npy")

    # --- Evaluation du modele baseline (entraine sur PlantVillage) sur PlantDoc ---
    print("\n=== EVALUATION DU DOMAIN GAP (baseline PlantVillage -> PlantDoc) ===")
    embedding_dim = embeddings.shape[1]
    n_classes = split_info["n_classes"]

    clf = nn.Linear(embedding_dim, n_classes)
    clf.load_state_dict(torch.load("models/linear_probe_best.pt"))
    clf.eval()

    X_plantdoc = torch.tensor(embeddings, dtype=torch.float32)
    y_plantdoc = np.array(all_labels)

    with torch.no_grad():
        logits = clf(X_plantdoc)
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_plantdoc, preds)
    f1_macro = f1_score(y_plantdoc, preds, average="macro")

    print(f"\nAccuracy sur PlantDoc (terrain): {acc:.4f}")
    print(f"F1 macro sur PlantDoc (terrain): {f1_macro:.4f}")
    print(f"\n(Rappel - performance sur PlantVillage test: accuracy=0.9876, f1_macro=0.9838)")
    print(f"\nCHUTE DE PERFORMANCE (domain gap): {(0.9876 - acc) * 100:.1f} points d'accuracy")

    unique_test_classes = sorted(set(y_plantdoc.tolist()))
    present_names = [labels_names[i] for i in unique_test_classes]

    print("\n=== Rapport detaille par classe (PlantDoc) ===")
    print(classification_report(y_plantdoc, preds, labels=unique_test_classes, target_names=present_names, zero_division=0))

if __name__ == "__main__":
    main()
