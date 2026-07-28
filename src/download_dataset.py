"""
Script de telechargement et exploration du dataset PlantVillage.
Semaine 1 - Etape 1 : Data engineering
"""

from datasets import load_dataset

def main():
    print("Telechargement du dataset PlantVillage (peut prendre quelques minutes)...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")

    print("\n=== Apercu du dataset ===")
    print(dataset)

    train = dataset["train"]
    print(f"\nNombre d'images (train): {len(train)}")
    print(f"Colonnes disponibles: {train.column_names}")

    if "label" in train.column_names:
        labels = train.features["label"].names
        print(f"\nNombre de classes: {len(labels)}")
        print("Classes disponibles:")
        for i, name in enumerate(labels):
            print(f"  {i}: {name}")
    else:
        print("\nATTENTION: pas de colonne label trouvee.")

    print("\nPremier exemple pour inspection:")
    example = train[0]
    for key, value in example.items():
        if key == "image":
            print(f"  image: {value}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
