# Modeles - etat au 2026-07-29

## MODELE OFFICIEL (le seul a utiliser)
- PRODUCTION_MODEL.pt : tete MLP, seed=42 pre-enregistre, sans selection a posteriori
  Score: PlantDoc test acc=68.35%, F1=56.42% | PlantVillage acc=97.52%
  Indices de split traces dans data/production_*.npy

## Modeles de reference valides (mais pas le modele final)
- linear_probe_best.pt : baseline PlantVillage-only, sert a mesurer le domain gap (37.46%)
- unified_model_fp32.pt / unified_model_int8.pt : PRODUCTION_MODEL.pt + DINOv2, pour export mobile

## Modeles obsoletes (archives/archive_obsolete/, NE PAS UTILISER)
Contamines par des fuites methodologiques corrigees depuis (voir git log pour le detail de l'audit).
