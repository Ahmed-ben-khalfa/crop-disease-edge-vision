"""
Calcul des prototypes de classe (moyenne des embeddings par classe) + calibration
d'un seuil de detection "hors distribution" (OOD), base sur la distance aux prototypes.
"""

import numpy as np
import json

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    labels_names = split_info["labels_names"]
    n_classes = split_info["n_classes"]

    print("Chargement des embeddings d'entrainement...")
    X_pv_train = np.load("data/train_embeddings.npy")
    y_pv_train = np.load("data/train_labels.npy")
    X_pd_full = np.load("data/plantdoc_embeddings.npy")
    y_pd_full = np.load("data/plantdoc_labels.npy")

    X_all = np.concatenate([X_pv_train, X_pd_full], axis=0)
    y_all = np.concatenate([y_pv_train, y_pd_full], axis=0)

    print("Calcul des prototypes (moyenne des embeddings par classe)...")
    prototypes = np.zeros((n_classes, X_all.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = y_all == c
        if mask.sum() > 0:
            prototypes[c] = X_all[mask].mean(axis=0)

    np.save("models/class_prototypes.npy", prototypes)
    print(f"Prototypes sauvegardes: models/class_prototypes.npy {prototypes.shape}")

    print("\nCalibration du seuil de detection hors-distribution...")
    X_val = np.load("data/val_embeddings.npy")
    y_val = np.load("data/val_labels.npy")

    def min_distance_to_prototypes(embedding, prototypes):
        dists = np.linalg.norm(prototypes - embedding, axis=1)
        return dists.min()

    in_dist_distances = np.array([
        min_distance_to_prototypes(X_val[i], prototypes) for i in range(len(X_val))
    ])

    print(f"Distances (donnees connues, val set): moyenne={in_dist_distances.mean():.2f}, "
          f"p95={np.percentile(in_dist_distances, 95):.2f}, p99={np.percentile(in_dist_distances, 99):.2f}")

    threshold = float(np.percentile(in_dist_distances, 99))
    print(f"\nSeuil OOD retenu (99e percentile des distances connues): {threshold:.2f}")

    with open("models/ood_threshold.json", "w") as f:
        json.dump({"threshold": threshold, "method": "min_distance_to_class_prototype_99th_percentile"}, f, indent=2)
    print("Seuil sauvegarde: models/ood_threshold.json")

    print("\n(Rappel: la photo de feuille de chene testee manuellement a donne une prediction")
    print("a 71% de confiance sur 'Grape_Black_rot' - le seuil de distance ci-dessus servira")
    print("a flaguer ce type de cas comme 'hors distribution probable' independamment du softmax.)")

if __name__ == "__main__":
    main()
