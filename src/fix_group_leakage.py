"""
Correction de la fuite au niveau image: regroupement des crops par photo source
avant le split train/test, pour eviter que deux crops de la meme image
se retrouvent de part et d'autre du split.
"""

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit
import json

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

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    name_to_id = {name: i for i, name in enumerate(labels_names)}

    print("Chargement de PlantDoc...")
    plantdoc = load_dataset("agyaatcoder/PlantDoc")

    # Reconstruction des groupes (image source) DANS LE MEME ORDRE que l'extraction originale
    groups = []
    labels_check = []
    global_image_counter = 0
    for split_name in ["train", "test"]:
        for example in plantdoc[split_name]:
            objects = example["objects"]
            has_valid_crop = False
            for bbox, category in zip(objects["bbox"], objects["category"]):
                if category not in CATEGORY_MAPPING:
                    continue
                mapped_name = CATEGORY_MAPPING[category]
                if mapped_name not in name_to_id:
                    continue
                groups.append(f"{split_name}_{global_image_counter}")
                labels_check.append(name_to_id[mapped_name])
                has_valid_crop = True
            global_image_counter += 1

    groups = np.array(groups)
    labels_check = np.array(labels_check)

    # Verification de coherence avec les embeddings existants
    y_pd_full = np.load("data/plantdoc_labels.npy")
    print(f"\nVerification: {len(groups)} groupes reconstruits vs {len(y_pd_full)} labels existants")
    assert len(groups) == len(y_pd_full), "MISMATCH - l'ordre ne correspond pas, ne pas continuer"
    assert np.array_equal(labels_check, y_pd_full), "MISMATCH labels - l'ordre ne correspond pas"
    print("Coherence confirmee: on peut reutiliser les embeddings existants avec ces groupes.")

    n_unique_images = len(np.unique(groups))
    print(f"\nNombre de crops total: {len(groups)}")
    print(f"Nombre d'images sources uniques: {n_unique_images}")
    print(f"Moyenne de crops par image: {len(groups)/n_unique_images:.2f}")

    # Split GROUPE (par image source), 70% train / 30% test - aucune image partagee
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    idx_train, idx_test = next(gss.split(np.zeros(len(groups)), y_pd_full, groups=groups))

    # Verification stricte: aucun groupe partage
    train_groups = set(groups[idx_train])
    test_groups = set(groups[idx_test])
    overlap = train_groups & test_groups
    print(f"\nVerification anti-fuite: {len(overlap)} images partagees entre train et test (doit etre 0)")
    assert len(overlap) == 0, "FUITE DETECTEE - le split groupe a un bug"

    print(f"\nNouveau split (par image, sans fuite):")
    print(f"  Train: {len(idx_train)} crops")
    print(f"  Test: {len(idx_test)} crops")

    np.save("data/plantdoc_groupsplit_train_idx.npy", idx_train)
    np.save("data/plantdoc_groupsplit_test_idx.npy", idx_test)
    print("\nIndices sauvegardes: data/plantdoc_groupsplit_train_idx.npy / _test_idx.npy")
    print("\nCOMPARAISON avec l'ancien split (par crop, avec fuite potentielle):")
    print("Ancien test: 2673 crops (possiblement contamine par des images partagees avec train)")
    print(f"Nouveau test: {len(idx_test)} crops (garanti sans image partagee avec train)")

if __name__ == "__main__":
    main()
