# 🌾 Roadmap Complet — Edge Vision Foundation Model pour Maladies des Cultures

> **Objectif final** : Un modèle de vision par ordinateur capable de diagnostiquer les maladies des plantes (+ leur sévérité) directement sur un smartphone, hors-ligne, en temps réel — à partir d'une simple photo de feuille.
> 
> **Pourquoi ce projet est puissant pour ton portfolio** : il combine 4 compétences très recherchées en 2026 — *foundation models / self-supervised learning (DINOv2)*, *few-shot learning*, *quantization / edge AI*, et *MLOps de bout en bout (dataset → modèle → app mobile)*. C'est exactement le genre de projet qui démontre une compréhension "full-stack ML".

---

## 📋 Table des matières

1. [Vue d'ensemble & concepts clés](#1-vue-densemble--concepts-clés)
2. [Prérequis & stack technique](#2-prérequis--stack-technique)
3. [Architecture du système](#3-architecture-du-système)
4. [Semaines 1-2 : Datasets & Baseline](#4-semaines-1-2--datasets--baseline)
5. [Semaines 3-4 : Robustesse terrain + Sévérité](#5-semaines-3-4--robustesse-terrain--sévérité)
6. [Semaines 5-6 : Quantization & Export mobile](#6-semaines-5-6--quantization--export-mobile)
7. [Semaines 7-8 : Évaluation de robustesse](#7-semaines-7-8--évaluation-de-robustesse)
8. [Livrables finaux & portfolio](#8-livrables-finaux--portfolio)
9. [Ressources d'apprentissage](#9-ressources-dapprentissage)
10. [Pièges courants à éviter](#10-pièges-courants-à-éviter)
11. [Checklist finale avant entretien](#11-checklist-finale-avant-entretien)

---

## 1. Vue d'ensemble & concepts clés

### 1.1 Le pipeline en une phrase

```
📷 Photo de feuille → 🧠 Vision Foundation Model (DINOv2/ViT) → 
⚠️ Classification maladie + sévérité → 📱 Inférence on-device (offline, quantifiée INT8)
```

### 1.2 Les 4 piliers techniques à maîtriser

| Pilier                 | Concept                                                        | Pourquoi c'est important                                                                                       |
| ---------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Foundation Model**   | DINOv2 (self-supervised ViT pré-entraîné par Meta)             | Apprend des représentations visuelles génériques sans labels → transférable à n'importe quelle culture/maladie |
| **Few-shot learning**  | Adapter le modèle avec très peu d'exemples par classe          | Les maladies rares n'ont pas des milliers d'images labellisées                                                 |
| **Robustesse terrain** | Généraliser à des photos "in-the-wild" (lumière, angle, bruit) | Un modèle qui marche en labo mais pas au champ ne sert à rien                                                  |
| **Edge deployment**    | Quantization INT8 + export mobile (TFLite/CoreML/ONNX)         | Le fermier n'a pas de GPU ni de connexion internet fiable                                                      |

### 1.3 Ce que tu vas savoir faire à la fin

- Construire et nettoyer un dataset de vision multi-source
- Fine-tuner un foundation model (DINOv2/ViT) avec des têtes de classification légères
- Implémenter du few-shot learning (prototypical networks, linear probing, LoRA)
- Ajouter une tête de régression/classification ordinale pour la sévérité
- Quantifier un modèle en INT8 (post-training quantization + QAT)
- Exporter vers TFLite / ONNX / Core ML et faire tourner l'inférence sur mobile
- Évaluer la robustesse (corruptions, domain shift, distribution réelle vs labo)

---

## 2. Prérequis & stack technique

### 2.1 Connaissances préalables recommandées

- Python solide (numpy, classes, décorateurs)
- Bases de deep learning (CNN, transformers, backprop, loss functions)
- Notions de PyTorch (Dataset/DataLoader, nn.Module, training loop)
- Git/GitHub

*Si tu ne les as pas encore à 100%, ce n'est pas grave — tu vas les consolider en semaine 1-2 en pratiquant directement sur le projet.*

### 2.2 Stack technique complète

**Langage & ML**

- Python 3.10+
- PyTorch 2.x + torchvision
- Hugging Face `transformers` + `timm` (pour DINOv2 et ViT pré-entraînés)
- `peft` (LoRA pour fine-tuning léger)
- `albumentations` (augmentations réalistes terrain)

**Data & expérimentation**

- `pandas`, `numpy`
- `Weights & Biases` (W&B) ou `MLflow` pour tracker les expériences
- Label Studio ou CVAT (si tu dois labelliser)
- DVC (versioning des datasets, optionnel mais très pro)

**Quantization & export**

- `torch.quantization` (PTQ + QAT natif PyTorch)
- `onnx` + `onnxruntime`
- `tensorflow` + `tflite` (si tu vises Android)
- `coremltools` (si tu vises iOS)

**Mobile / déploiement**

- Android : Kotlin + TensorFlow Lite / MediaPipe
- iOS : Swift + Core ML
- Alternative rapide : **React Native + `onnxruntime-react-native`** ou une app de démo simple

**Infra**

- Google Colab Pro / Kaggle (GPU gratuit/pas cher au début)
- AWS/GCP/Lambda Labs si besoin de plus de puissance (RTX 4090 ou A10 suffisent largement)

---

## 3. Architecture du système

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE D'ENTRAÎNEMENT                   │
│                                                                   │
│  Datasets (PlantVillage, PlantDoc, iNaturalist crops, terrain)   │
│         │                                                        │
│         ▼                                                        │
│  Nettoyage + Augmentation (albumentations : flou, luminosité,   │
│  occlusion, angles variés, arrière-plans réels)                 │
│         │                                                        │
│         ▼                                                        │
│  DINOv2 (backbone gelé ou LoRA) ──► Feature embeddings (768/1024d)│
│         │                                                        │
│         ├──► Tête 1 : Classification maladie (few-shot / linear  │
│         │    probing / prototypical network)                     │
│         │                                                        │
│         └──► Tête 2 : Sévérité (régression ordinale 0-4,         │
│              ex: sain / léger / modéré / sévère / critique)      │
│         │                                                        │
│         ▼                                                        │
│  Distillation / Quantization (INT8, QAT)                         │
│         │                                                        │
│         ▼                                                        │
│  Export ONNX → TFLite (Android) / Core ML (iOS)                  │
│         │                                                        │
│         ▼                                                        │
│              📱 APP MOBILE (inférence 100% offline)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Semaines 1-2 : Datasets & Baseline

### 🎯 Objectif de la phase

Construire un dataset propre et entraîner un premier modèle baseline fonctionnel (même imparfait) pour valider le pipeline de bout en bout.

### Semaine 1 — Data engineering

**Jour 1-2 : Recherche et collecte de données**

- Télécharge et explore ces datasets publics :
  - **PlantVillage** (54k+ images, 38 classes, conditions labo — bon point de départ mais peu réaliste)
  - **PlantDoc** (~2600 images, conditions réelles — essentiel pour la robustesse)
  - **iNaturalist** (filtré sur les cultures qui t'intéressent, via l'API)
  - **Kaggle "Plant Pathology 2020/2021"** (pommes, avec labels de sévérité — parfait pour la tête 2)
- Choisis 2-3 cultures cibles pour rester focus (ex : tomate, pomme, maïs) plutôt que de viser "toutes les cultures" — un scope raisonnable montre de la maturité produit.

**Jour 3-4 : Exploration & nettoyage**

- Notebook EDA : distribution des classes, résolution des images, doublons (utilise `imagehash` pour détecter les quasi-doublons entre datasets)
- Corrige le déséquilibre de classes (visualise avec des histogrammes)
- Split rigoureux : train / val / test — **attention au data leakage** si plusieurs photos viennent de la même plante/feuille (split par *plante*, pas par image)

**Jour 5 : Pipeline de données PyTorch**

- Écris une classe `Dataset` custom qui unifie les différentes sources avec un mapping de labels cohérent
- Mets en place `albumentations` avec des augmentations réalistes :
  - Variation de luminosité/contraste (simule le soleil/l'ombre au champ)
  - Flou de mouvement léger (main qui tremble)
  - Rotation et crop aléatoire (angle de prise de vue)
  - Ajout de bruit/JPEG compression (qualité caméra variable)

### Semaine 2 — Baseline model

**Jour 1-2 : Setup DINOv2**

- Charge `facebook/dinov2-base` via `transformers` ou `torch.hub`
- Comprends la différence entre :
  - **Linear probing** (backbone gelé, tu entraînes juste une couche linéaire sur les embeddings) → rapide, bon premier test
  - **Fine-tuning complet** (coûteux, risque d'overfitting avec peu de données)
  - **LoRA** (le compromis idéal — tu entraînes seulement de petites matrices de rang faible injectées dans les couches d'attention)

**Jour 3-4 : Entraînement baseline**

- Entraîne un classifieur linéaire simple sur les embeddings DINOv2 (frozen backbone)
- Log toutes tes expériences avec W&B (loss, accuracy, F1 par classe, confusion matrix)
- Métriques à suivre dès le départ : **accuracy globale**, **F1 macro** (important car classes déséquilibrées), **matrice de confusion**

**Jour 5 : Validation du pipeline complet**

- Vérifie que tu peux faire : image → preprocessing → embedding → prédiction → affichage résultat, de bout en bout dans un notebook
- Documente ton baseline dans un `README.md` avec les métriques obtenues (ex: 85% accuracy sur PlantVillage — normal, c'est facile, le vrai test viendra en semaine 3-4)

### 📦 Livrable de fin de phase

- Repo GitHub avec structure propre (`data/`, `src/`, `notebooks/`, `configs/`)
- Dataset unifié versionné (DVC ou juste un script de téléchargement reproductible)
- Modèle baseline + rapport de métriques dans W&B
- README expliquant le choix des datasets et du split

---

## 5. Semaines 3-4 : Robustesse terrain + Sévérité

### 🎯 Objectif de la phase

Passer d'un modèle "qui marche en labo" à un modèle robuste aux conditions réelles, et ajouter la prédiction de sévérité.

### Semaine 3 — Généralisation "in-the-wild"

**Jour 1-2 : Diagnostic du gap labo → terrain**

- Entraîne sur PlantVillage (labo), teste sur PlantDoc (terrain) → observe la chute de performance (c'est normal et attendu, documente-le, c'est un excellent point à raconter en entretien : *"j'ai mesuré un domain gap de X%, voici comment je l'ai réduit"*)

**Jour 3-4 : Few-shot learning**

- Implémente une **Prototypical Network** : pour chaque classe, calcule le "prototype" (moyenne des embeddings) à partir de quelques exemples (5-shot, 10-shot), puis classe par distance au prototype le plus proche
- Compare avec le linear probing classique
- Teste aussi une approche **LoRA fine-tuning** sur DINOv2 pour voir si ça bat le few-shot pur

**Jour 5 : Data augmentation avancée + mélange des domaines**

- Mixe PlantVillage + PlantDoc + tes propres photos (prends des photos de plantes réelles si possible, même saines, pour du negative sampling)
- Technique de **style transfer léger / mixup** entre images labo et terrain pour réduire le domain gap

### Semaine 4 — Tête de sévérité

**Jour 1-2 : Design de la tête de sévérité**

- Deux approches possibles :
  1. **Classification ordinale** (0=sain, 1=léger, 2=modéré, 3=sévère, 4=critique) avec une loss qui pénalise plus les erreurs "loin" (ex: CORAL loss ou soft ordinal encoding)
  2. **Régression continue** (score 0-100% de surface foliaire atteinte) si le dataset le permet (ex: Plant Pathology 2020 a ce genre de labels)
- Architecture : partage le backbone DINOv2, deux têtes séparées (multi-task learning), avec une loss combinée : `L = L_classification + λ * L_severity`

**Jour 3-4 : Entraînement multi-tâche**

- Attention à l'équilibrage des losses (λ à tuner — sinon une tâche domine l'autre)
- Valide séparément les deux têtes (accuracy pour la maladie, MAE/QWK — Quadratic Weighted Kappa — pour la sévérité, métrique standard en classification ordinale médicale)

**Jour 5 : Ablation study**

- Compare : tête de sévérité seule vs multi-tâche vs modèles séparés
- Documente laquelle est la meilleure et pourquoi (le multi-tâche partage souvent des features utiles entre les deux tâches → meilleure généralisation)

### 📦 Livrable de fin de phase

- Modèle multi-tâche (maladie + sévérité) entraîné avec few-shot learning
- Rapport comparatif : linear probing vs prototypical network vs LoRA
- Métriques de robustesse : accuracy labo vs terrain, QWK pour la sévérité
- Visualisations (t-SNE/UMAP des embeddings par classe — très visuel pour un portfolio)

---

## 6. Semaines 5-6 : Quantization & Export mobile

### 🎯 Objectif de la phase

Transformer ton modèle de recherche (lourd, FP32) en un modèle léger et rapide qui tourne sur smartphone.

### Semaine 5 — Compression du modèle

**Jour 1-2 : Comprendre les options de compression**

- **Distillation** : entraîne un petit modèle (ex: MobileViT, EfficientNet-Lite, ou un ViT-Tiny) à imiter les prédictions de ton gros modèle DINOv2 (le "teacher") → c'est souvent nécessaire car DINOv2-base est trop lourd pour du mobile pur
- **Pruning** : supprime les poids/neurones peu importants (moins prioritaire que la distillation ici)
- **Quantization** : réduis la précision numérique (FP32 → INT8), le sujet central de cette phase

**Jour 3-4 : Distillation teacher-student**

- Choisis un student léger : MobileViT-XS ou EfficientFormer (bon compromis vitesse/précision pour mobile)
- Entraîne le student avec une loss de distillation (KL divergence sur les logits + éventuellement matching des features intermédiaires)
- Valide que le student garde >90% de la performance du teacher

**Jour 5 : Premiers tests de quantization**

- **Post-Training Quantization (PTQ)** avec `torch.quantization` : calibre sur un petit set de données représentatif, convertis en INT8
- Mesure la perte de précision (souvent 1-3% de chute avec PTQ, acceptable)

### Semaine 6 — Export et intégration mobile

**Jour 1-2 : Quantization-Aware Training (QAT)**

- Si PTQ perd trop de précision, fais du QAT : simule la quantization pendant l'entraînement (fake quantization nodes) pour que le modèle s'adapte
- Compare PTQ vs QAT sur tes métriques

**Jour 3 : Export ONNX**

- Convertis le modèle PyTorch quantifié en ONNX
- Valide l'équivalence numérique (les prédictions ONNX doivent matcher PyTorch à epsilon près)

**Jour 4 : Export plateforme finale**

- **Android** : ONNX → TensorFlow → TFLite (via `onnx-tf` ou directement `torch.onnx` puis conversion), teste avec `tflite_runtime`
- **iOS** : ONNX → Core ML via `coremltools`
- Mesure : taille du modèle (Mo), latence d'inférence (ms) sur CPU mobile simulé, RAM utilisée

**Jour 5 : App de démo mobile**

- Construis une app minimale (Android Kotlin ou Flutter ou React Native) qui :
  - Ouvre la caméra
  - Capture une image
  - Fait tourner le modèle TFLite/CoreML localement (100% offline)
  - Affiche : maladie détectée + niveau de sévérité + confiance
- Même une app "quick and dirty" mais qui marche vaut mieux qu'une app parfaite jamais finie

### 📦 Livrable de fin de phase

- Modèle final quantifié INT8 (taille cible : <20-30 Mo)
- Benchmark comparatif : taille, latence, accuracy (FP32 vs INT8, teacher vs student)
- App mobile fonctionnelle avec vidéo de démo (**élément le plus impressionnant pour un recruteur**)
- Fichiers exportés : `.onnx`, `.tflite` et/ou `.mlmodel`

---

## 7. Semaines 7-8 : Évaluation de robustesse

### 🎯 Objectif de la phase

Prouver rigoureusement que ton modèle est fiable en conditions réelles, pas juste sur un test set académique.

### Semaine 7 — Tests de robustesse systématiques

**Jour 1-2 : Corruption benchmarks**

- Applique des corruptions standardisées façon **ImageNet-C** : flou gaussien, bruit, changements de luminosité/contraste, compression JPEG à différents niveaux, pixelisation
- Mesure la dégradation de l'accuracy pour chaque type et intensité de corruption → génère un graphique "accuracy vs sévérité de corruption"

**Jour 3-4 : Test sur données out-of-distribution (OOD)**

- Teste sur des feuilles saines de plantes qui ne sont dans aucune classe (ton modèle doit savoir dire "je ne sais pas" plutôt que d'halluciner une maladie)
- Implémente une détection OOD simple (seuil sur la confiance softmax, ou distance aux prototypes en embedding space)

**Jour 5 : Test sur tes propres photos terrain**

- Si possible, prends 50-100 photos réelles (ou demande à des proches/agriculteurs locaux) dans des conditions variées (matin/soir, soleil/nuage, différents fonds)
- C'est ton "vrai" test set — le plus précieux pour ton portfolio car personne d'autre ne l'a

### Semaine 8 — Analyse finale, explicabilité et packaging

**Jour 1-2 : Explicabilité (XAI)**

- Génère des **Grad-CAM** ou **attention maps** (les ViT s'y prêtent très bien via leurs poids d'attention) pour montrer *où* le modèle regarde sur la feuille
- Ça permet de détecter si le modèle triche (ex : il regarde l'arrière-plan au lieu de la feuille — signe de biais dataset)

**Jour 3 : Analyse d'échecs (error analysis)**

- Regroupe les cas d'échec par catégorie (mauvais angle ? confusion entre 2 maladies visuellement proches ? sévérité mal calibrée ?)
- Documente 5-10 exemples d'échecs avec ton interprétation — **ça montre une vraie maturité ML en entretien**

**Jour 4 : Rapport technique final**

- Rédige un rapport (style paper court, 4-6 pages) : problème, données, méthode, résultats, limites, travaux futurs
- Inclus tous les graphiques : courbes d'apprentissage, matrices de confusion, robustesse aux corruptions, comparaison des architectures testées

**Jour 5 : Polish final du repo et de la démo**

- README GitHub impeccable avec badges, GIF de démo de l'app, instructions d'installation reproductibles
- Vidéo de démo de 1-2 minutes (à publier sur LinkedIn/portfolio)
- Nettoie le code, ajoute des tests unitaires basiques, formatte avec `black`/`ruff`

### 📦 Livrable de fin de phase

- Rapport de robustesse complet avec visualisations
- Attention maps / Grad-CAM pour l'explicabilité
- Rapport technique final (PDF ou Markdown)
- Repo GitHub "portfolio-ready"

---

## 8. Livrables finaux & portfolio

À la fin des 8 semaines, tu dois avoir :

- [ ] **Un repo GitHub public** bien structuré et documenté
- [ ] **Un rapport technique** (PDF/Markdown) expliquant toute la démarche
- [ ] **Une app mobile de démo** fonctionnelle (même simple)
- [ ] **Une vidéo de démo** (1-2 min) montrant l'app en action sur une vraie feuille
- [ ] **Un post LinkedIn/portfolio** résumant le projet avec les chiffres clés (accuracy, taille du modèle, latence)
- [ ] **Des visualisations soignées** : matrices de confusion, t-SNE des embeddings, Grad-CAM, courbes de robustesse

### Comment présenter ça en entretien

Structure ton pitch en 4 points :

1. **Le problème** : diagnostiquer les maladies au champ, sans connexion, en temps réel
2. **Le défi technique principal** que tu as résolu (ex : "réduire le domain gap labo→terrain de X% à Y%" ou "compresser un ViT de 300 Mo à 15 Mo en perdant seulement 2% d'accuracy")
3. **Les trade-offs** que tu as dû arbitrer (précision vs vitesse, few-shot vs full fine-tuning)
4. **Ce que tu ferais différemment** avec plus de temps/données (montre ta capacité d'auto-critique)

---

## 9. Ressources d'apprentissage

**Foundation models & DINOv2**

- Paper original DINOv2 (Meta AI, 2023)
- Documentation `timm` et `transformers` sur les backbones ViT

**Few-shot learning**

- Paper "Prototypical Networks for Few-shot Learning" (Snell et al.)
- Cours en ligne sur le meta-learning (ex: Stanford CS330)

**Quantization**

- Documentation officielle PyTorch sur `torch.quantization`
- Blog posts Hugging Face sur l'optimisation de modèles pour l'edge

**Robustesse**

- Paper "Benchmarking Neural Network Robustness to Common Corruptions" (Hendrycks & Dietterich) — la base d'ImageNet-C

**Datasets**

- PlantVillage, PlantDoc, Kaggle Plant Pathology 2020/2021, iNaturalist

*(Conseil : cherche les versions les plus récentes de ces papers/docs au moment où tu commences, les librairies évoluent vite.)*

---

## 10. Pièges courants à éviter

- ❌ **Data leakage** : mêmes plantes dans train et test → accuracy gonflée artificiellement
- ❌ **Overfitting sur PlantVillage** : ce dataset est "trop facile" (fonds uniformes) → toujours valider sur PlantDoc ou du réel
- ❌ **Ignorer le déséquilibre de classes** : utilise F1 macro et pas juste l'accuracy globale
- ❌ **Quantifier trop tôt** : optimise d'abord la précision, quantifie en dernier
- ❌ **Sous-estimer le temps de l'app mobile** : l'intégration mobile prend souvent plus de temps que prévu, garde de la marge
- ❌ **Vouloir couvrir toutes les cultures/maladies** : reste focus sur 2-3 cultures pour un projet de qualité plutôt que superficiel

---

## 11. Checklist finale avant entretien

- [ ] Je peux expliquer pourquoi j'ai choisi DINOv2 plutôt qu'un CNN classique
- [ ] Je peux expliquer la différence entre PTQ et QAT et pourquoi j'ai choisi l'un ou l'autre
- [ ] Je peux donner des chiffres précis (accuracy, taille modèle, latence, QWK)
- [ ] Je peux montrer une démo vidéo fonctionnelle
- [ ] Je peux parler d'au moins un échec/limite et de comment je l'ai analysé
- [ ] Mon code est propre, testé et sur GitHub avec un bon README

---

**Bon courage 🚀 — ce projet, bien exécuté, est largement suffisant pour démontrer une compétence ML end-to-end de niveau professionnel. Prends ton temps sur les fondamentaux (semaines 1-2), le reste suivra plus vite.**
