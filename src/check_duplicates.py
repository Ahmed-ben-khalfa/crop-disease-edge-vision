"""
Detection des doublons/quasi-doublons entre train et test.
Semaine 1 - Etape 2 : Exploration et nettoyage (suite)
"""

from datasets import load_dataset
import imagehash

def main():
    print("Chargement du dataset (depuis le cache local)...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    train = dataset["train"]
    test = dataset["test"]

    print(f"\nCalcul des hash perceptuels sur {len(train)} images train...")
    print("(cela peut prendre 1-2 minutes)")

    train_hashes = {}
    for i, img in enumerate(train["image"]):
        h = imagehash.phash(img)
        train_hashes[str(h)] = i
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(train)} traitees...")

    print(f"\nCalcul des hash perceptuels sur {len(test)} images test...")
    duplicates = 0
    for i, img in enumerate(test["image"]):
        h = imagehash.phash(img)
        if str(h) in train_hashes:
            duplicates += 1

    print(f"\n=== RESULTAT ===")
    print(f"Images test identiques (hash exact) a une image train: {duplicates}")
    print(f"Pourcentage du test set concerne: {100 * duplicates / len(test):.2f}%")

if __name__ == "__main__":
    main()
