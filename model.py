"""
Model architecture for oilseed plant classification using transfer learning.
Uses MobileNet (V2 or V3) from torchvision — 4 classes only.
"""
import ssl
import torch
import torch.nn as nn
import torchvision.models as models
import config


def _apply_ssl_workaround():
    """
    Contournement pour macOS / Python.org : certificat SSL non reconnu
    lors du téléchargement des poids pré-entraînés PyTorch.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def get_model(model_name=None, pretrained=True, num_classes=None):
    """
    Create a model for oilseed plant classification (4 classes).

    Args:
        model_name: 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of output classes (default: 4)

    Returns:
        PyTorch model
    """
    if model_name is None:
        model_name = config.MODEL_NAME
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    # Contournement SSL pour le téléchargement des poids (macOS / CERTIFICATE_VERIFY_FAILED)
    if pretrained:
        _apply_ssl_workaround()

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # Replace classifier: MobileNetV2 has classifier = Sequential(0: Dropout, 1: Linear)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1" if pretrained else None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: "
            "mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, "
            "resnet50, efficientnet_b0, efficientnet_b3"
        )

    return model


def freeze_backbone(model, freeze=True, model_name=None):
    """
    Freeze or unfreeze the backbone (feature extractor), keep classifier trainable.

    Args:
        model: PyTorch model
        freeze: If True, freeze backbone; if False, unfreeze
        model_name: Optional; if None, uses config.MODEL_NAME

    Returns:
        Model with frozen/unfrozen backbone
    """
    name = (model_name or config.MODEL_NAME).lower()

    if "mobilenet_v2" in name:
        for name_param, param in model.named_parameters():
            if "classifier" not in name_param:
                param.requires_grad = not freeze
    elif "mobilenet_v3" in name:
        for name_param, param in model.named_parameters():
            if "classifier" not in name_param:
                param.requires_grad = not freeze
    elif "resnet" in name:
        for name_param, param in model.named_parameters():
            if "fc" not in name_param:
                param.requires_grad = not freeze
    elif "efficientnet" in name:
        for name_param, param in model.named_parameters():
            if "classifier" not in name_param:
                param.requires_grad = not freeze

    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model()
    print(f"Model: {config.MODEL_NAME}")
    print(f"Number of trainable parameters: {count_parameters(model):,}")

    x = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (1, {config.NUM_CLASSES})")
