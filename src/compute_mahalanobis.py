"""
Distance de Mahalanobis pour la detection hors-distribution (Lee et al. 2018,
"A Simple Unified Framework for Detecting Out-of-Distribution Samples").
Plus rigoureuse que la distance euclidienne: prend en compte la variance
naturelle de chaque classe plutot qu'une simple distance au centre.
"""

import numpy as np
import json

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    n_classes = split_info["n_classes"]

    prototypes = np.load("models/class_prototypes.npy")

    print("Chargement des embeddings d'entrainement...")
    X_pv_train = np.load("data/train_embeddings.npy")
    y_pv_train = np.load("data/train_labels.npy")
    X_pd_full = np.load("data/plantdoc_embeddings.npy")
    y_pd_full = np.load("data/plantdoc_labels.npy")

    X_all = np.concatenate([X_pv_train, X_pd_full], axis=0)
    y_all = np.concatenate([y_pv_train, y_pd_full], axis=0)

    print(f"Calcul de la covariance partagee sur {len(X_all)} echantillons ({X_all.shape[1]} dimensions)...")

    centered = X_all - prototypes[y_all]
    cov = (centered.T @ centered) / len(centered)

    epsilon = 1e-3 * np.trace(cov) / cov.shape[0]
    cov_reg = cov + epsilon * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov_reg)

    np.save("models/mahalanobis_cov_inv.npy", cov_inv)
    print(f"Matrice de covariance inversee sauvegardee: models/mahalanobis_cov_inv.npy {cov_inv.shape}")

    def mahalanobis_min_distance(x, prototypes, cov_inv):
        diffs = prototypes - x
        dists_sq = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
        return np.sqrt(np.maximum(dists_sq, 0)).min()

    print("\nCalcul des distances de Mahalanobis (validation PlantVillage + test PlantDoc)...")
    X_val_pv = np.load("data/val_embeddings.npy")
    X_pd_test = np.load("data/plantdoc_test_embeddings.npy")

    dist_val = np.array([mahalanobis_min_distance(X_val_pv[i], prototypes, cov_inv) for i in range(len(X_val_pv))])
    dist_pd_test = np.array([mahalanobis_min_distance(X_pd_test[i], prototypes, cov_inv) for i in range(len(X_pd_test))])

    print(f"PlantVillage val (Mahalanobis): moyenne={dist_val.mean():.2f}, p99={np.percentile(dist_val,99):.2f}, max={dist_val.max():.2f}")
    print(f"PlantDoc test (Mahalanobis): moyenne={dist_pd_test.mean():.2f}, p99={np.percentile(dist_pd_test,99):.2f}, max={dist_pd_test.max():.2f}")

    combined = np.concatenate([dist_val, dist_pd_test])
    new_threshold = float(np.percentile(combined, 99))
    print(f"\nNouveau seuil Mahalanobis (p99, combine labo+terrain): {new_threshold:.2f}")

    with open("models/ood_threshold.json", "w") as f:
        json.dump({
            "threshold": new_threshold,
            "method": "mahalanobis_distance_shared_covariance_99th_percentile"
        }, f, indent=2)
    print("Seuil sauvegarde: models/ood_threshold.json")
    print("\nATTENTION: le code de l'app doit maintenant utiliser la distance de Mahalanobis")
    print("(pas la distance euclidienne) pour que ce seuil soit coherent. Mise a jour a suivre.")

if __name__ == "__main__":
    main()
