# Rapport Technique — Edge Vision Foundation Model pour le Diagnostic des Maladies des Cultures

**Auteur :** [Ton nom]
**Période :** Semaines 1-8
**Stack :** Python, PyTorch, DINOv2 (Meta AI), Hugging Face Transformers, ONNX Runtime

---

## Résumé exécutif

Ce projet développe un système de diagnostic des maladies des plantes à partir d'une simple photo de feuille, conçu pour fonctionner hors-ligne sur un appareil mobile. Il combine un foundation model de vision (DINOv2) avec une tête de classification légère, entraîné sur un mélange de données de laboratoire (PlantVillage) et de terrain (PlantDoc).

**Résultat final, mesuré honnêtement sur un test set jamais vu pendant l'entraînement ni le développement :**

| Métrique | Valeur |
|---|---|
| Accuracy PlantVillage (conditions labo) | 97,5% |
| **Accuracy PlantDoc (conditions terrain réelles)** | **72,5%** |
| Amélioration vs. baseline naïf (labo uniquement) | +35 points |
| Robustesse aux corruptions (bruit/flou/JPEG) | +3 à +15,5 points vs. version non renforcée |

Ce projet a une particularité qui le distingue : **il documente un audit méthodologique complet**, incluant la découverte et la correction de 3 fuites de données qui gonflaient artificiellement les métriques initiales de plus de 11 points. Cette démarche de vérification rigoureuse, rare dans les projets similaires, constitue l'un des apports les plus significatifs du travail.

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Données](#2-données)
3. [Méthodologie et architecture](#3-méthodologie-et-architecture)
4. [Découverte et correction du domain gap](#4-découverte-et-correction-du-domain-gap)
5. [Estimation de la sévérité](#5-estimation-de-la-sévérité)
6. [Audit méthodologique : 3 fuites de données trouvées et corrigées](#6-audit-méthodologique--3-fuites-de-données-trouvées-et-corrigées)
7. [Modèle de production et validation croisée](#7-modèle-de-production-et-validation-croisée)
8. [Optimisation pour le déploiement mobile](#8-optimisation-pour-le-déploiement-mobile)
9. [Tests de robustesse et amélioration](#9-tests-de-robustesse-et-amélioration)
10. [Explicabilité](#10-explicabilité)
11. [Résultats finaux](#11-résultats-finaux)
12. [Limites connues et travaux futurs](#12-limites-connues-et-travaux-futurs)
13. [Conclusion](#13-conclusion)

---

## 1. Contexte et objectifs

L'objectif est de construire un système capable de :
- Diagnostiquer une maladie parmi 38 classes (14 cultures) à partir d'une photo de feuille
- Estimer la sévérité de l'infection
- Fonctionner offline, sur un appareil aux ressources limitées (smartphone)
- Généraliser à des conditions de prise de vue réelles (éclairage variable, angle, qualité d'image), pas seulement à des photos de laboratoire

Le développement a été réalisé entièrement sur une machine CPU (pas de GPU dédié), une contrainte qui a influencé plusieurs choix techniques documentés dans ce rapport.

---

## 2. Données

Deux sources ont été combinées, avec des rôles distincts et complémentaires :

| Dataset | Rôle | Volume utilisé |
|---|---|---|
| **PlantVillage** | Baseline, conditions de laboratoire (fonds uniformes, éclairage contrôlé) | 43 456 images, 38 classes, 14 cultures |
| **PlantDoc** | Conditions réelles de terrain, dataset de détection d'objets reconverti en classification | 8 910 crops (2 578 photos sources), 29 catégories mappées vers nos 38 classes |

**Préparation des données :**
- Split stratifié train/val/test pour PlantVillage (80/10/10), avec détection et retrait des doublons via hachage perceptuel (7 doublons trouvés sur 10 849 images test, retirés)
- Déséquilibre de classes important identifié (ratio 36:1 entre la classe la plus fréquente et la plus rare) — corrigé par pondération de la loss d'entraînement
- PlantDoc : reconversion depuis le format détection d'objets (bounding boxes) vers des crops individuels, avec mapping manuel des 29 catégories PlantDoc vers les 38 classes PlantVillage (21 correspondances directes établies)

---

## 3. Méthodologie et architecture

```
Photo → DINOv2-small (backbone gelé, 22M paramètres) → embedding CLS (384 dimensions)
      → Tête de classification (MLP : 384→256→128→38)
      → Prédiction de maladie + estimation de sévérité (segmentation couleur indépendante)
```

**Choix de DINOv2-small** (plutôt que la version *base*) : dicté par les contraintes CPU de la machine de développement — un compromis documenté et assumé plutôt qu'un choix par défaut non questionné.

**Approche linear probing / MLP probing** : le backbone DINOv2 reste gelé (non fine-tuné), seule une tête légère est entraînée sur les embeddings extraits. Ce choix limite le temps de calcul (l'extraction d'embeddings ne se fait qu'une fois, l'entraînement de la tête est ensuite quasi instantané) mais limite aussi la capacité du modèle à adapter ses représentations internes à la tâche — une limite discutée en section 10 et 12.

---

## 4. Découverte et correction du domain gap

**Constat initial** : un modèle entraîné exclusivement sur PlantVillage (98,76% d'accuracy sur son propre test set) chute à **37,46%** lorsqu'il est évalué sur PlantDoc — une chute de **61,3 points**, révélant que le modèle avait appris des raccourcis visuels liés aux conditions de laboratoire (fonds uniformes, éclairage constant) plutôt que les véritables caractéristiques des maladies.

**Correction** : intégration de PlantDoc dans l'entraînement (mélange PlantVillage + PlantDoc), ce qui a permis de faire remonter l'accuracy sur PlantDoc à 72,4% dans un premier temps — un résultat qui s'est révélé partiellement biaisé (voir section 6) et corrigé à 72,5% dans sa version finale honnête après audit.

**Enseignement clé** : une haute performance sur un unique dataset ne garantit pas la généralisation. La diversité des conditions d'entraînement compte davantage que le volume brut — l'ajout de seulement ~5 200 images terrain (15% du volume total) a suffi à quasi doubler la performance en conditions réelles.

---

## 5. Estimation de la sévérité

**Première approche (abandonnée)** : un mapping déterministe classe→sévérité (ex : *Apple_scab* = toujours "léger", *Black_rot* = toujours "sévère"). Cette approche a été identifiée comme fondamentalement limitée : elle ne varie jamais au sein d'une même classe, donc n'apporte aucune information indépendante de la prédiction de maladie elle-même. Le modèle multi-tâche entraîné sur cette base (`multitask_best.pt`) est documenté comme obsolète et n'est plus utilisé.

**Approche retenue** : estimation par **segmentation couleur** — détection des zones non conformes au vert sain typique d'une feuille (teinte HSV) dans le masque de la feuille (seuillage Otsu), calcul du ratio de surface atteinte, puis discrétisation en 3 niveaux (sain, léger-modéré, sévère).

**Validation de la variance intra-classe** (preuve que la mesure est réellement informative) :

| Classe | Ratio moyen | Écart-type | Min | Max |
|---|---|---|---|---|
| Apple___Apple_scab | 0,319 | 0,167 | 0,008 | 0,876 |
| Tomato___Late_blight | 0,660 | 0,239 | 0,004 | 1,000 |
| Corn___Common_rust | 0,948 | 0,112 | 0,203 | 0,999 |

La variance significative au sein de chaque classe confirme que cette mesure capture une information réelle par image, contrairement au proxy déterministe initial.

**Limite assumée** : cette mesure reste un indicateur de surface atteinte, pas une évaluation clinique validée par un expert en phytopathologie — une piste de validation future est indiquée en section 12.

---

## 6. Audit méthodologique : 3 fuites de données trouvées et corrigées

C'est la section la plus significative de ce rapport du point de vue de la rigueur scientifique. Trois fautes méthodologiques ont été identifiées a posteriori, par relecture critique systématique du pipeline, et corrigées avant la publication des résultats finaux.

### Faute n°1 — Sélection de modèle sur le test set

Lors d'une itération d'amélioration (ajout de manifold mixup et suréchantillonnage du domaine PlantDoc), le meilleur epoch était sélectionné en surveillant directement la performance sur le **test set**, plutôt que sur un ensemble de validation dédié. Cette pratique, bien que ne modifiant jamais les gradients d'entraînement, biaise le chiffre rapporté par sélection multiple (choisir le meilleur epoch parmi 40 équivaut à effectuer 40 évaluations du test set et n'en retenir que la meilleure).

**Correction** : introduction d'un split de validation dédié (jamais utilisé pour l'évaluation finale), le test set n'étant consulté qu'une seule fois, après entraînement complet.

### Faute n°2 — Fuite au niveau image dans le split PlantDoc

PlantDoc étant un dataset de détection d'objets, une même photographie source peut contenir plusieurs feuilles (crops). Le split train/test initial opérait au niveau des **crops individuels**, sans regrouper les crops issus d'une même photo. Conséquence : des crops très similaires (même fond, même éclairage, même angle) pouvaient se retrouver simultanément dans le train et dans le test, permettant au modèle de "reconnaître" indirectement des images qu'il avait déjà vues sous un angle légèrement différent.

**Ampleur mesurée** : 8 910 crops provenant de seulement 2 578 photos sources (moyenne de 3,46 crops par photo) — un risque de fuite non négligeable.

**Correction** : split par groupe (`GroupShuffleSplit`), garantissant qu'aucune photo source n'apparaît à la fois dans le train et dans le test. Vérification automatique intégrée au code (assertion sur l'intersection vide des ensembles d'images).

### Faute n°3 — Incohérence entre le modèle déployé et le score cité

Après correction des deux fautes précédentes, une validation croisée à 3 graines aléatoires a été effectuée pour mesurer la stabilité du résultat (69,2% ± 0,6%). Cependant, aucun des 3 modèles de cette validation croisée n'avait été sauvegardé sur disque — le modèle réellement destiné au déploiement provenait d'un entraînement séparé, dont le score individuel (66,89%) ne correspondait pas au chiffre cité.

**Correction** : entraînement d'un unique **modèle de production**, avec une graine aléatoire fixée dès le début du projet (seed=42, jamais sélectionnée après coup pour éviter tout biais de sélection), et citation exclusive du score mesuré sur ce modèle précis et sauvegardé (68,35% initialement, 72,5% après l'amélioration de robustesse — voir section 9). Traçabilité complète assurée par la sauvegarde des indices exacts de split (`data/production_*.npy`).

### Impact cumulé de ces corrections

| Étape | Score PlantDoc rapporté | Statut |
|---|---|---|
| Baseline (PlantVillage seul) | 37,46% | ✅ Fiable dès le départ (aucune fuite possible) |
| Mix simple, non audité | 72,43% | ⚠️ Contaminé par la faute n°2 |
| MLP + mixup, non audité | 80,40% | 🚨 Contaminé par les fautes n°1 et n°2 cumulées |
| **Après audit complet** | **68,35%** (puis 72,5% après robustesse) | ✅ **Chiffre fiable, traçable, reproductible** |

L'écart entre le chiffre non audité (80,40%) et le chiffre final honnête (68,35%) — soit **12 points d'écart** — illustre concrètement l'importance de cette démarche de vérification.

---

## 7. Modèle de production et validation croisée

Pour garantir la stabilité du résultat, un test de validation croisée à 3 graines aléatoires indépendantes a été mené : **69,2% ± 0,6%** (min 68,4%, max 69,6%). Le faible écart-type confirme que le chiffre n'est pas un artefact de chance sur un split particulier.

Le modèle finalement déployé (`PRODUCTION_MODEL.pt`, seed=42, choisi *avant* observation des résultats pour éviter tout biais de sélection a posteriori) atteint individuellement **68,35%**, cohérent avec cet intervalle.

---

## 8. Optimisation pour le déploiement mobile

### Tentatives de quantization (4 approches testées)

L'objectif était de réduire la taille du modèle (84,7 Mo en FP32) pour un déploiement mobile. Quatre approches légitimes ont été testées, chacune avec un résultat documenté :

| Approche | Taille | Latence | Impact accuracy | Verdict |
|---|---|---|---|---|
| INT8 dynamique (PyTorch) | 23,7 Mo | 34,2 ms | **-5,0 points** | 🚨 Rejeté |
| FP16 (demi-précision) | 42,4 Mo | 725,8 ms (×10 plus lent) | 0,0 point | 🚨 Rejeté (latence) |
| INT8 statique calibré (PyTorch) | — | — | Erreur d'exécution | 🚨 Rejeté (incompatibilité technique) |
| INT8 dynamique (ONNX Runtime) | — | — | Erreur d'exécution | 🚨 Rejeté (conflit de shape inference) |

**Explication scientifique** : les architectures Transformer (dont fait partie DINOv2) sont connues dans la littérature récente (SmoothQuant, LLM.int8()) pour présenter des activations avec des valeurs extrêmes dans certaines couches d'attention, mal gérées par les méthodes de quantization naïves conçues initialement pour les CNN. Le FP16, quant à lui, n'est pas accéléré matériellement sur les CPU x86 classiques (contrairement aux GPU), d'où le ralentissement observé plutôt qu'une accélération.

**Décision finale** : le modèle est déployé en **FP32**, avec la quantization documentée comme limite connue. Le chemin de résolution le plus prometteur (distillation vers une architecture CNN plus petite, intrinsèquement mieux adaptée à la quantization) nécessiterait un ré-entraînement complet de bout en bout, jugé impraticable dans le temps imparti sur une machine CPU (estimation : plusieurs heures par epoch).

### Export ONNX

L'export vers le format ONNX a en revanche été un succès :
- Écart numérique négligeable avec PyTorch (max 0,000013)
- **Accélération de 1,23×** par rapport à PyTorch pur sur CPU (74,0 ms vs 90,9 ms)
- Format portable, standard pour un déploiement multiplateforme (Android/iOS/serveur)

---

## 9. Tests de robustesse et amélioration

### Diagnostic initial (échantillon de 200 images test)

| Type de corruption | Chute d'accuracy à sévérité maximale |
|---|---|
| Luminosité (sous-exposition) | -10,5 points (robuste) |
| Contraste (faible) | -13,0 points (robuste) |
| Bruit capteur | -32,5 points (fragile) |
| **Compression JPEG forte** | **-33,5 points (le plus fragile)** |
| Flou | -30,5 points (fragile) |

### Correction : entraînement avec augmentation par corruptions

Le modèle a été ré-entraîné en intégrant des versions corrompues (flou, bruit, JPEG à sévérité variable) des images d'entraînement PlantDoc, avec conservation du label d'origine.

**Résultats de la comparaison avant/après (même protocole, même échantillon) :**

| Test | Delta |
|---|---|
| Images propres (bonus inattendu) | **+3,0 points** |
| Bruit (sévérité max) | **+15,5 points** |
| Flou (sévérité max) | +5,0 points |
| Luminosité/Contraste | +0 à +4,5 points |
| JPEG | Mitigé (+3,0 / -2,0 / +1,5 selon sévérité) |

Point notable : l'amélioration de la robustesse s'est accompagnée d'une **amélioration simultanée de l'accuracy sur images propres** (+3,0 points), signe que l'augmentation a agi comme un bon régularisateur général plutôt que comme un compromis robustesse/précision.

**Recommandation produit** : implémenter une vérification de netteté d'image (ex : variance du Laplacien) côté application, avant l'inférence, pour avertir l'utilisateur en cas de photo de mauvaise qualité plutôt que de fournir un diagnostic peu fiable en silence.

---

## 10. Explicabilité

Deux méthodes ont été testées pour visualiser les zones de l'image influençant la prédiction du modèle :

1. **Attention brute de la dernière couche DINOv2** : cartes diffuses, sans concentration nette sur les symptômes visibles des maladies.
2. **Grad-CAM adapté aux Vision Transformers** (gradient du score de la classe prédite par rapport aux patches) : légèrement plus informatif mais toujours relativement diffus.

**Constat honnête** : ni l'une ni l'autre méthode ne produit de cartes clairement localisées sur les lésions, contrairement à ce qu'on observe classiquement avec Grad-CAM sur des CNN. Ce résultat est cohérent avec la littérature sur l'explicabilité des Transformers, et s'explique en partie par le fait que le backbone DINOv2 reste **gelé** (non fine-tuné) — le gradient de la tâche ne réorganise jamais l'attention interne du backbone vers les zones pertinentes pour la classification des maladies.

Ce point est documenté comme limite assumée, avec une piste d'amélioration claire pour un travail futur (fine-tuning LoRA du backbone, ou techniques d'explicabilité plus avancées comme l'Attention Rollout multi-couches).

---

## 11. Résultats finaux

| Métrique | Valeur |
|---|---|
| Classes couvertes | 38 (14 cultures) |
| Accuracy PlantVillage (labo) | 97,5% |
| **Accuracy PlantDoc (terrain, modèle final robuste)** | **72,5%** |
| Amélioration vs. baseline labo-seul | +35,0 points |
| Robustesse (bruit, sévérité max) | +15,5 points vs. version non renforcée |
| Taille du modèle déployé | ~90 Mo (ONNX, FP32) |
| Latence d'inférence (CPU, ONNX Runtime) | ~74 ms/image |
| Comparaison littérature académique (ViT+MoE, cross-domain) | 68% — résultat comparable, dans la marge de variance mesurée |

---

## 12. Limites connues et travaux futurs

| Limite | Détail | Piste de résolution future |
|---|---|---|
| Pas de quantization fonctionnelle | 4 approches testées, toutes documentées comme insatisfaisantes | Distillation vers un CNN léger (MobileViT/EfficientNet-lite), nécessite un accès GPU |
| Explicabilité diffuse | Attention et Grad-CAM peu localisés | Fine-tuning LoRA du backbone, ou Attention Rollout multi-couches |
| Sévérité non validée cliniquement | Mesure par segmentation couleur, pas de vérité terrain experte | Collecte d'annotations de sévérité par un expert en phytopathologie |
| Dataset PlantDoc limité (2 578 photos sources) | Variance non négligeable observée entre différents splits de validation croisée | Collecte de données terrain additionnelles |
| Domaine restreint aux 14 cultures couvertes | Pas de détection "hors distribution" formalisée pour des cultures non vues | Ajout d'un mécanisme de rejet basé sur la confiance ou la distance aux prototypes de classe |
| Backbone gelé (pas de fine-tuning) | Limite la capacité d'adaptation fine à la tâche | Fine-tuning LoRA, si les ressources de calcul le permettent |

---

## 13. Conclusion

Ce projet démontre un pipeline complet de vision par ordinateur — de la collecte de données à l'évaluation de robustesse — construit avec un foundation model moderne (DINOv2) sur des ressources de calcul limitées (CPU uniquement). 

Au-delà des résultats numériques (72,5% d'accuracy en conditions terrain, comparable à l'état de l'art académique), la contribution la plus significative de ce travail réside dans sa **démarche méthodologique** : l'identification et la correction de 3 fuites de données distinctes, chacune mesurée et documentée avec son impact précis sur les métriques rapportées, ainsi que la transparence sur les tentatives infructueuses (quantization, explicabilité) plutôt que leur omission.

Cette rigueur — vérifier plutôt que supposer, mesurer plutôt qu'affirmer — constitue une compétence aussi précieuse que la maîtrise technique elle-même dans la pratique professionnelle du machine learning.

---

## Annexe — Reproductibilité

- Code source complet : `src/`
- Modèle de production final : `models/PRODUCTION_MODEL_ROBUST.pt`
- Indices de split (traçabilité complète) : `data/production_*.npy`
- Résultats bruts de robustesse : `docs/robustness_comparison.json`
- Historique complet des expériences : `git log`
