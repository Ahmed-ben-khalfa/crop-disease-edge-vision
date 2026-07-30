# 🌾 Edge Vision Foundation Model — Diagnostic des Maladies des Cultures

> Diagnostic de maladies foliaires à partir d'une simple photo, conçu pour fonctionner hors-ligne sur mobile. Construit avec un foundation model de vision (DINOv2), entraîné sur des données de laboratoire ET de terrain, avec un audit méthodologique complet.

## 🎯 Résultats clés

| Métrique | Valeur |
|---|---|
| Classes couvertes | 38 maladies (14 cultures) |
| Accuracy en conditions terrain (PlantDoc) | **72,5%** |
| Amélioration vs. baseline labo-seul | **+35,0 points** |
| Comparaison littérature académique (ViT+MoE) | 68% — performance comparable |
| Robustesse aux corruptions (bruit) | +15,5 points après renforcement |

📄 **[Lire le rapport technique complet](docs/RAPPORT_FINAL.md)**

## 🔍 Ce qui distingue ce projet

La plupart des projets de ce type s'arrêtent au premier chiffre encourageant. Celui-ci va plus loin : un **audit méthodologique a posteriori a révélé 3 fuites de données** qui gonflaient artificiellement les métriques de plus de 11 points — chacune identifiée, mesurée, et corrigée avant de publier un résultat final honnête et reproductible.

- ✅ Sélection de modèle biaisée par le test set → corrigée (split de validation dédié)
- ✅ Fuite d'image dans le split PlantDoc (crops de la même photo dans train et test) → corrigée (split par groupe)
- ✅ Incohérence entre le modèle déployé et le score cité → corrigée (modèle de production unique, tracé)

4 approches de quantization ont aussi été testées et **honnêtement documentées comme non concluantes** pour cette architecture Vision Transformer sur cette machine — plutôt que de prétendre à un succès non vérifié.

## 🏗️ Architecture

```
Photo → DINOv2-small (backbone gelé) → embedding CLS (384d)
      → Tête MLP (384→256→128→38)
      → Prédiction de maladie + sévérité (segmentation couleur indépendante)
```

## 📂 Structure du repo

```
├── src/           # Scripts (dataset, entraînement, évaluation, quantization, robustesse)
├── docs/          # Rapport final, roadmap, résultats et visualisations
├── models/        # Modèle de production final (voir models/README.md)
├── data/          # Métadonnées de split (traçabilité complète)
```

## 🚀 Utiliser le modèle

```bash
python src/predict.py chemin/vers/photo.jpg
```

## 🛠️ Stack technique

Python · PyTorch · DINOv2 (Meta AI) · Hugging Face Transformers · ONNX Runtime · scikit-learn

## 📊 Datasets utilisés

- [PlantVillage](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) — conditions de laboratoire
- [PlantDoc](https://huggingface.co/datasets/agyaatcoder/PlantDoc) — conditions de terrain réelles

---

*Projet développé sur 8 semaines, entièrement sur CPU (sans GPU dédié). Voir le [rapport complet](docs/RAPPORT_FINAL.md) pour la méthodologie détaillée, l'audit des fuites de données, et les limites assumées.*
