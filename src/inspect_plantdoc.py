"""
Inspection de la structure du dataset PlantDoc.
Semaine 3 - Etape 1 : Exploration de PlantDoc (donnees terrain).
"""

from datasets import load_dataset

def main():
    print("Chargement de PlantDoc...")
    dataset = load_dataset("agyaatcoder/PlantDoc")

    print("\n=== Apercu du dataset ===")
    print(dataset)

    train = dataset["train"]
    print(f"\nColonnes disponibles: {train.column_names}")

    print("\n=== Premier exemple ===")
    example = train[0]
    for key, value in example.items():
        if key == "image":
            print(f"  image: {value}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
