"""
Semaine 5-6 - Export ONNX du modele de production.
Format portable standard pour le deploiement (Android/iOS/serveur).
Bonus: tentative de quantization via ONNX Runtime (outils souvent plus matures
que PyTorch eager mode pour les architectures Transformer).
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from transformers import AutoModel, AutoImageProcessor

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
    def __init__(self, n_classes, embedding_dim=384):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.head = MLPHead(embedding_dim, n_classes)
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)

def main():
    with open("data/split_info.json", "r") as f:
        split_info = json.load(f)
    n_classes = split_info["n_classes"]

    print("Chargement du modele de production...")
    model = UnifiedModel(n_classes)
    model.head.load_state_dict(torch.load("models/PRODUCTION_MODEL.pt", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print("\nExport vers ONNX...")
    onnx_path = "models/production_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"Export termine: {onnx_path} ({onnx_size_mb:.1f} Mo)")

    # --- Validation numerique: PyTorch vs ONNX doivent donner les memes resultats ---
    print("\nValidation numerique (PyTorch vs ONNX)...")
    import onnxruntime as ort

    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = ort_session.run(None, {"pixel_values": dummy_input.numpy()})[0]

    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Difference maximale entre PyTorch et ONNX: {max_diff:.6f}")
    print(f"Verdict: {'IDENTIQUE (aux erreurs numeriques pres)' if max_diff < 1e-3 else 'DIVERGENCE DETECTEE'}")

    # --- Mesure de latence ONNX Runtime vs PyTorch ---
    print("\nMesure de latence...")
    def measure_pytorch(n=15):
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        times = []
        with torch.no_grad():
            for _ in range(n):
                start = time.time()
                _ = model(dummy_input)
                times.append((time.time() - start) * 1000)
        return sum(times) / len(times)

    def measure_onnx(n=15):
        inp = {"pixel_values": dummy_input.numpy()}
        for _ in range(3):
            _ = ort_session.run(None, inp)
        times = []
        for _ in range(n):
            start = time.time()
            _ = ort_session.run(None, inp)
            times.append((time.time() - start) * 1000)
        return sum(times) / len(times)

    lat_pytorch = measure_pytorch()
    lat_onnx = measure_onnx()
    print(f"PyTorch FP32: {lat_pytorch:.1f} ms")
    print(f"ONNX Runtime FP32: {lat_onnx:.1f} ms ({lat_pytorch/lat_onnx:.2f}x)")

    print("\n" + "="*70)
    print("=== BONUS: tentative de quantization dynamique via ONNX Runtime ===")
    print("="*70)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = "models/production_model_int8.onnx"
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QInt8
        )
        quant_size_mb = os.path.getsize(quantized_path) / (1024 ** 2)
        print(f"Quantization ONNX Runtime reussie: {quantized_path} ({quant_size_mb:.1f} Mo)")
        print(f"Reduction de taille: {(1 - quant_size_mb/onnx_size_mb)*100:.1f}%")

        ort_session_int8 = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
        onnx_int8_output = ort_session_int8.run(None, {"pixel_values": dummy_input.numpy()})[0]
        diff_int8 = np.abs(pytorch_output - onnx_int8_output).max()
        print(f"Difference max vs PyTorch FP32 (sortie brute, un seul echantillon aleatoire): {diff_int8:.4f}")
        print("(Un test complet d'accuracy sur le vrai test set sera fait dans le script suivant)")

    except Exception as e:
        print(f"ECHEC de la quantization ONNX Runtime: {type(e).__name__}: {e}")
        print("On continue avec le modele ONNX FP32 uniquement.")

if __name__ == "__main__":
    main()
