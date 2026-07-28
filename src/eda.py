"""
Analyse exploratoire (EDA) du dataset PlantVillage.
Semaine 1 - Etape 2 : Exploration et nettoyage
"""

from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

def main():
    print("Chargement du dataset (depuis le cache local)...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    train = dataset["train"]
    labels_names = train.features["label"].names

    # Comptage par classe
    counts = Counter(train["label"])
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])

    print("\n=== Distribution des classes (train) ===")
    print(f"{'Classe':<55} {'Nb images':>10}")
    print("-" * 66)
    for label_id, count in sorted_counts:
        print(f"{labels_names[label_id]:<55} {count:>10}")

    min_class = sorted_counts[0]
    max_class = sorted_counts[-1]
    print(f"\nClasse la plus rare : {labels_names[min_class[0]]} ({min_class[1]} images)")
    print(f"Classe la plus frequente : {labels_names[max_class[0]]} ({max_class[1]} images)")
    print(f"Ratio desequilibre (max/min): {max_class[1] / min_class[1]:.1f}x")

    # Graphique
    names = [labels_names[i] for i, _ in sorted_counts]
    values = [c for _, c in sorted_counts]

    plt.figure(figsize=(10, 14))
    plt.barh(names, values, color="#3b7a57")
    plt.xlabel("Nombre d'images")
    plt.title("Distribution des classes - PlantVillage (train)")
    plt.tight_layout()
    plt.savefig("docs/class_distribution.png", dpi=150)
    print("\nGraphique sauvegarde dans docs/class_distribution.png")

if __name__ == "__main__":
    main()
