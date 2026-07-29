"""
Semaine 5 - Etape 1 : Construction d'un modele unifie (DINOv2 + classifieur)
pour pouvoir le quantifier et l'exporter en un seul bloc.
Mesure de la taille et de la latence AVANT optimisation (baseline a battre).
"""

import torch
import torch.nn as nn
import time
import os
import json
from transformers import AutoModel

MODEL_NAME = "facebook/dinov2-small"

class MLPHead(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class UnifiedModel(nn.Module):
    """DINOv2 backbone + tete de classification, en un seul module exportable."""
    def __init__(self, n_classes, embedding_dim=384):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.head = MLPHead(embedding_dim, n_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_embedding)
        return logits

def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def measure_latency(model, dummy_input, n_runs=20):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(dummy_input)
            times.append((time.time() - start) * 1000)  # ms
    return sum(times) / len(times), min(times), max(times)

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    n_classes = split_info["n_classes"]

    print("Construction du modele unifie (DINOv2-small + MLP)...")
    model = UnifiedModel(n_classes)
    model.head.load_state_dict(torch.load("models/PRODUCTION_MODEL.pt", map_location="cpu"))
    model.eval()

    print("\n=== BASELINE (avant optimisation) ===")
    size_mb = get_model_size_mb(model)
    print(f"Taille du modele (FP32): {size_mb:.1f} Mo")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de parametres: {n_params:,}")

    # Image standard DINOv2: 224x224x3
    dummy_input = torch.randn(1, 3, 224, 224)
    avg_ms, min_ms, max_ms = measure_latency(model, dummy_input)
    print(f"\nLatence d'inference (CPU, batch=1, 224x224):")
    print(f"  Moyenne: {avg_ms:.1f} ms")
    print(f"  Min: {min_ms:.1f} ms | Max: {max_ms:.1f} ms")
    print(f"  Debit: {1000/avg_ms:.1f} images/seconde")

    # Sauvegarde du modele unifie complet pour la suite (quantization/export)
    torch.save(model.state_dict(), "models/unified_model_fp32.pt")
    print(f"\nModele unifie sauvegarde: models/unified_model_fp32.pt")

    print("\n=== OBJECTIFS POUR LA QUANTIZATION ===")
    print(f"Taille actuelle: {size_mb:.1f} Mo -> objectif: <{size_mb/3:.0f} Mo (INT8, ~4x plus petit)")
    print(f"Latence actuelle: {avg_ms:.1f} ms -> objectif: reduction significative")
    print("(Rappel: objectif produit = <30 Mo, temps reel sur mobile)")

if __name__ == "__main__":
    main()
