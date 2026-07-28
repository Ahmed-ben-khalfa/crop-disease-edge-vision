"""
Liste de toutes les categories presentes dans PlantDoc.
Semaine 3 - Etape 1 (suite) : Preparation du mapping de classes.
"""

from datasets import load_dataset
from collections import Counter

def main():
    print("Chargement de PlantDoc (depuis le cache)...")
    dataset = load_dataset("agyaatcoder/PlantDoc")

    all_categories = Counter()
    for split_name in ["train", "test"]:
        for example in dataset[split_name]:
            for cat in example["objects"]["category"]:
                all_categories[cat] += 1

    print(f"\n=== {len(all_categories)} categories uniques dans PlantDoc ===\n")
    for cat, count in sorted(all_categories.items()):
        print(f"  {cat:<40} {count:>5} occurrences")

if __name__ == "__main__":
    main()
