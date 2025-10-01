import torch
import sys

print("--- Verificação de GPU ---")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} está ativa.")
else:
    print("ERRO CRÍTICO: CUDA não está disponível.")
    sys.exit(1)