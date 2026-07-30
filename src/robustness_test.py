"""
Semaine 7 - Tests de robustesse aux corruptions (facon ImageNet-C).
Mesure la degradation de l'accuracy face a des conditions realistes degradees:
flou, bruit, luminosite/contraste, compression JPEG.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import io
from PIL import Image, ImageFilter, ImageEnhance
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset
from sklearn.metrics import accuracy_score

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

def corrupt_blur(img, severity):
    radius = [1, 2.5, 5][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def corrupt_noise(img, severity):
    arr = np.array(img).astype(np.float32)
    std = [10, 25, 50][severity - 1]
    noise = np.random.default_rng(0).normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def corrupt_brightness(img, severity):
    factor = [0.6, 0.4, 0.2][severity - 1]
    return ImageEnhance.Brightness(img).enhance(factor)

def corrupt_contrast(img, severity):
    factor = [0.5, 0.3, 0.15][severity - 1]
    return ImageEnhance.Contrast(img).enhance(factor)

def corrupt_jpeg(img, severity):
    quality = [40, 15, 5][severity - 1]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

CORRUPTIONS = {
    "blur (flou de mouvement/mise au point)": corrupt_blur,
    "noise (bruit capteur)": corrupt_noise,
    "brightness (sous-exposition)": corrupt_brightness,
    "contrast (faible contraste)": corrupt_contrast,
    "jpeg (compression forte)": corrupt_jpeg,
}

def rebuild_test_crops(name_to_id, indices):
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
    sel_crops = [all_crops[i] for i in indices]
    sel_labels = [all_labels[i] for i in indices]
    return sel_crops, sel_labels

def run_inference(model, processor, crops, batch_size=16):
    preds = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i+batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            logits = model(inputs["pixel_values"])
            preds.extend(torch.argmax(logits, dim=1).tolist())
    return preds

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Chargement du modele de production...")
    model = UnifiedModel(n_classes)
    model.head.load_state_dict(torch.load("models/PRODUCTION_MODEL.pt", map_location="cpu"))
    model.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Reconstruction d'un echantillon du test set (200 images)...")
    test_idx = np.load("data/production_test_idx.npy")
    rng = np.random.default_rng(42)
    sample_pos = rng.choice(len(test_idx), size=min(200, len(test_idx)), replace=False)
    sample_group_idx = test_idx[sample_pos]
    test_crops, test_labels = rebuild_test_crops(name_to_id, sample_group_idx)
    test_labels = np.array(test_labels)
    print(f"Echantillon: {len(test_crops)} images")

    print("\nBaseline (images propres, sans corruption)...")
    preds_clean = run_inference(model, processor, test_crops)
    acc_clean = accuracy_score(test_labels, preds_clean)
    print(f"Accuracy sans corruption: {acc_clean:.4f}")

    results = {"clean": acc_clean}

    print("\n" + "="*70)
    print("TESTS DE ROBUSTESSE PAR TYPE DE CORRUPTION")
    print("="*70)

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        print(f"\n--- {corruption_name} ---")
        severity_accs = []
        for severity in [1, 2, 3]:
            corrupted_crops = [corruption_fn(img, severity) for img in test_crops]
            preds = run_inference(model, processor, corrupted_crops)
            acc = accuracy_score(test_labels, preds)
            severity_accs.append(acc)
            drop = (acc_clean - acc) * 100
            print(f"  Severite {severity}: accuracy={acc:.4f} (chute de {drop:.1f} points vs propre)")
        results[corruption_name] = severity_accs

    print("\n" + "="*70)
    print("=== TABLEAU RECAPITULATIF ===")
    print("="*70)
    print(f"Clean (reference): {acc_clean:.4f}")
    for name, accs in results.items():
        if name == "clean":
            continue
        print(f"{name:<45} Sev1={accs[0]:.4f}  Sev2={accs[1]:.4f}  Sev3={accs[2]:.4f}")

    with open("docs/robustness_results.json", "w") as f:
        json.dump({"clean_accuracy": acc_clean, "corruptions": results, "n_samples": len(test_crops)}, f, indent=2)
    print("\nResultats sauvegardes: docs/robustness_results.json")

    most_damaging = max(
        [(name, accs[0] - accs[2]) for name, accs in results.items() if name != "clean"],
        key=lambda x: x[1]
    )
    print(f"\nCorruption la plus dommageable (severite 1 -> 3): {most_damaging[0]} ({most_damaging[1]*100:.1f} points de chute)")

if __name__ == "__main__":
    main()
