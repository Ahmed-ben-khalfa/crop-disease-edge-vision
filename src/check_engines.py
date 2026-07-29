import torch
print("Moteurs de quantization supportes sur cette machine:")
print(torch.backends.quantized.supported_engines)
