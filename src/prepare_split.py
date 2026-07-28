"""
Nettoyage des doublons + split train/val/test + poids de classe.
Semaine 1 - Etape 2 (fin) : Preparation finale du dataset
"""

from datasets import load_dataset
import imagehash
import numpy as np
import json
from collections import Counter

def main():
    print("Chargement du dataset...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    train = dataset["train"]
    test = dataset["test"]
    labels_names = train.features["label"].names

    # --- 1. Retirer les doublons du test set ---
    print("\nRecalcul des hash pour identifier les doublons a retirer...")
    train_hashes = set()
    for img in train["image"]:
        train_hashes.add(str(imagehash.phash(img)))

    keep_indices = []
    for i, img in enumerate(test["image"]):
        h = str(imagehash.phash(img))
        if h not in train_hashes:
            keep_indices.append(i)

    test_clean = test.select(keep_indices)
    print(f"Test set nettoye: {len(test)} -> {len(test_clean)} images ({len(test) - len(test_clean)} doublons retires)")

    # --- 2. Split stratifie train/val (80/20) a partir du train ---
    print("\nCreation du split train/val stratifie...")
    labels_array = np.array(train["label"])
    train_idx = []
    val_idx = []

    rng = np.random.default_rng(42)
    for class_id in range(len(labels_names)):
        class_indices = np.where(labels_array == class_id)[0]
        rng.shuffle(class_indices)
        split_point = int(len(class_indices) * 0.8)
        train_idx.extend(class_indices[:split_point].tolist())
        val_idx.extend(class_indices[split_point:].tolist())

    print(f"Train final: {len(train_idx)} images")
    print(f"Validation: {len(val_idx)} images")
    print(f"Test: {len(test_clean)} images")

    # --- 3. Poids de classe (pour compenser le desequilibre) ---
    print("\nCalcul des poids de classe...")
    counts = Counter(labels_array[train_idx].tolist())
    total = sum(counts.values())
    n_classes = len(labels_names)

    class_weights = {}
    for class_id in range(n_classes):
        count = counts.get(class_id, 1)
        # Formule standard: poids inversement proportionnel a la frequence
        weight = total / (n_classes * count)
        class_weights[class_id] = round(weight, 4)

    print("Poids de classe calcules (top 5 plus eleves = classes rares):")
    sorted_weights = sorted(class_weights.items(), key=lambda x: -x[1])
    for class_id, w in sorted_weights[:5]:
        print(f"  {labels_names[class_id]}: poids={w}")

    # --- 4. Sauvegarde des indices et metadonnees ---
    split_info = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "class_weights": class_weights,
        "labels_names": labels_names,
        "n_classes": n_classes,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_clean),
    }

    with open("data/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("\nInfos de split sauvegardees dans data/split_info.json")
    print("\n=== RECAPITULATIF FINAL ===")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_clean)}")
    print(f"Classes: {n_classes}")
    print("Pret pour l'entrainement du modele baseline (semaine 2).")

if __name__ == "__main__":
    main()
