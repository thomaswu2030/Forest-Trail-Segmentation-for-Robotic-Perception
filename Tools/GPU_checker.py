# check if GPU could be applied in model training
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    # perform an operation on the GPU.
    x = torch.randn(100, 100, device="cuda")
    y = x @ x
    print("GPU calculation worked:", y.shape)