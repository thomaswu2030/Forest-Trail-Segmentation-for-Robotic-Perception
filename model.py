import torch
from torch import nn
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights,
)


# retain pretrained features and learn two output scores per pixel, other and trail.
def build_model(pretrained=True):
    if not pretrained:
        return lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2)
    # build with pretrained weight
    model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    # LR-ASPP has TWO prediction branches; replace both, retaining learned features.
    for name in ("low_classifier", "high_classifier"):
        previous = getattr(model.classifier, name)
        setattr(model.classifier, name, nn.Conv2d(previous.in_channels, 2, kernel_size=1))
    return model


# simply choose device. use cuda if available
def choose_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select --device cpu or check your PyTorch environment.")
    return torch.device(name)


# read a saved checkpoint
def load_checkpoint(path, device):
    # Load onto CPU first so a GPU-trained checkpoint can also be used on a CPU
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("classes") != ["other", "trail"]:
        raise ValueError("Unsupported checkpoint format or label mapping")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    # eval() selects inference behavior; callers separately disable gradient tracking.
    return model.to(device).eval(), checkpoint


if __name__ == "__main__":
    device = choose_device()
    model = build_model().to(device).eval()
    with torch.inference_mode():
        result = model(torch.randn(1, 3, 256, 448, device=device))["out"]
    print("Device:", device)
    print("Output:", result.shape, "(batch, other/trail, height, width)")
