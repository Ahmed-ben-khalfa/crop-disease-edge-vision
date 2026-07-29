"""
Estimation de severite REELLE par segmentation couleur des lesions.
Correction du point faible: severity independante de la classe.
Semaine 4 (correction) : vraie mesure de severite par image.
"""

import numpy as np
import cv2
from datasets import load_dataset
import json
import time

def estimate_lesion_ratio(pil_image):
    """
    Estime le ratio de surface foliaire atteinte via segmentation couleur.
    Logique: en HSV, le feuillage sain est dans une plage de vert specifique.
    Tout ce qui n'est pas "vert sain" a l'interieur de la feuille = lesion potentielle
    (jaune, brun, noir, taches).
    """
    img = np.array(pil_image.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Masque de la feuille entiere (vs fond) : on garde tout sauf le fond tres clair/uniforme
    # Approche simple: la feuille occupe la zone non-blanche/non-fond
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    leaf_mask = leaf_mask > 0

    leaf_area = leaf_mask.sum()
    if leaf_area < 100:
        # Fallback: toute l'image est consideree comme feuille
        leaf_mask = np.ones(gray.shape, dtype=bool)
        leaf_area = leaf_mask.sum()

    # Plage de "vert sain" en HSV (teinte verte, saturation/valeur raisonnables)
    lower_healthy = np.array([25, 40, 40])
    upper_healthy = np.array([90, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy) > 0

    # Zone malade = dans la feuille MAIS pas dans le vert sain
    diseased_mask = leaf_mask & (~healthy_mask)
    diseased_area = diseased_mask.sum()

    ratio = diseased_area / max(leaf_area, 1)
    return float(np.clip(ratio, 0.0, 1.0))

def ratio_to_severity(ratio):
    """Discretisation du ratio en 3 niveaux (bornes basees sur la litterature phytopathologique)."""
    if ratio < 0.05:
        return 0  # Sain / tres leger
    elif ratio < 0.25:
        return 1  # Leger a modere
    else:
        return 2  # Severe

def process_split(images, labels_names_ref=""):
    ratios = []
    severities = []
    n = len(images)
    start = time.time()
    for i, img in enumerate(images):
        ratio = estimate_lesion_ratio(img)
        ratios.append(ratio)
        severities.append(ratio_to_severity(ratio))
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - start
            print(f"  [{labels_names_ref}] {i}/{n} traitees ({i/elapsed:.0f} img/s)")
    return np.array(ratios), np.array(severities)

def main():
    print("Chargement du dataset PlantVillage...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    train_full = dataset["train"]

    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    train_idx = split_info["train_indices"]
    val_idx = split_info["val_indices"]

    print(f"\nCalcul de severite REELLE (segmentation) sur TRAIN ({len(train_idx)} images)...")
    train_images = [train_full[i]["image"] for i in train_idx]
    train_ratios, train_severities = process_split(train_images, "TRAIN")
    np.save("data/train_severity_ratios.npy", train_ratios)
    np.save("data/train_severity_labels.npy", train_severities)

    print(f"\nCalcul de severite REELLE sur VAL ({len(val_idx)} images)...")
    val_images = [train_full[i]["image"] for i in val_idx]
    val_ratios, val_severities = process_split(val_images, "VAL")
    np.save("data/val_severity_ratios.npy", val_ratios)
    np.save("data/val_severity_labels.npy", val_severities)

    print("\n=== VERIFICATION: la severite varie-t-elle DANS une meme classe ? ===")
    train_labels = np.array([train_full[i]["label"] for i in train_idx])
    labels_names = split_info["labels_names"]

    # On verifie sur 3 classes malades si le ratio varie (preuve que ce n'est pas juste un lookup)
    for class_name in ["Apple___Apple_scab", "Tomato___Late_blight", "Corn_(maize)___Common_rust_"]:
        class_id = labels_names.index(class_name)
        mask = train_labels == class_id
        class_ratios = train_ratios[mask]
        print(f"{class_name}: ratio moyen={class_ratios.mean():.3f}, std={class_ratios.std():.3f}, min={class_ratios.min():.3f}, max={class_ratios.max():.3f}")

    print("\nDistribution globale des severites (train):", np.bincount(train_severities))
    print("\nTermine. Fichiers sauvegardes dans data/*_severity_*.npy")

if __name__ == "__main__":
    main()
