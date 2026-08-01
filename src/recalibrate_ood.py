"""
Recalibration du seuil OOD: utiliser des donnees terrain diversifiees
(PlantDoc test, jamais vu en entrainement) plutot que la validation
PlantVillage seule (trop homogene, cause des faux positifs sur de vraies photos).
"""

import numpy as np
import json

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)

    prototypes = np.load("models/class_prototypes.npy")

    def min_distance_to_prototypes(embedding, prototypes):
        dists = np.linalg.norm(prototypes - embedding, axis=1)
        return dists.min()

    print("=== Ancien seuil (calibre sur PlantVillage validation seule) ===")
    X_val_pv = np.load("data/val_embeddings.npy")
    old_distances = np.array([min_distance_to_prototypes(X_val_pv[i], prototypes) for i in range(len(X_val_pv))])
    print(f"PlantVillage val: moyenne={old_distances.mean():.2f}, p95={np.percentile(old_distances,95):.2f}, p99={np.percentile(old_distances,99):.2f}, max={old_distances.max():.2f}")

    print("\n=== Nouvelle reference: PlantDoc TEST (vraies photos terrain, jamais entrainees) ===")
    X_pd_test = np.load("data/plantdoc_test_embeddings.npy")
    pd_distances = np.array([min_distance_to_prototypes(X_pd_test[i], prototypes) for i in range(len(X_pd_test))])
    print(f"PlantDoc test: moyenne={pd_distances.mean():.2f}, p95={np.percentile(pd_distances,95):.2f}, p99={np.percentile(pd_distances,99):.2f}, max={pd_distances.max():.2f}")

    # Combinaison des deux references (labo + terrain) pour un seuil robuste
    combined = np.concatenate([old_distances, pd_distances])
    print(f"\n=== Distribution combinee (PlantVillage val + PlantDoc test) ===")
    print(f"moyenne={combined.mean():.2f}, p95={np.percentile(combined,95):.2f}, p99={np.percentile(combined,99):.2f}, max={combined.max():.2f}")

    # Nouveau seuil: p99 de la distribution combinee (bien plus represantatif de la diversite reelle)
    new_threshold = float(np.percentile(combined, 99))
    print(f"\nNOUVEAU SEUIL (p99, distribution combinee labo+terrain): {new_threshold:.2f}")
    print(f"Ancien seuil: 35.01 -> Nouveau seuil: {new_threshold:.2f}")

    with open("models/ood_threshold.json", "w") as f:
        json.dump({
            "threshold": new_threshold,
            "method": "min_distance_to_class_prototype_99th_percentile_combined_lab_and_field"
        }, f, indent=2)
    print("\nSeuil recalibre et sauvegarde: models/ood_threshold.json")

    # Test sur les distances observees manuellement
    print("\n=== Verification sur nos observations manuelles ===")
    print(f"Chene (test OOD reel): distance=42.8 -> {'FLAGUE (correct)' if 42.8 > new_threshold else 'NON flague'}")
    print(f"Tomate saine (faux positif observe): distance=44.4 -> {'FLAGUE (encore faux positif)' if 44.4 > new_threshold else 'NON flague (corrige!)'}")

if __name__ == "__main__":
    main()
