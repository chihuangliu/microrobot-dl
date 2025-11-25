import torch.nn as nn
from torchvision import models


resnet_models = ["resnet18", "resnet34", "resnet50"]
alexnet_models = ["alexnet"]
vgg_models = ["vgg11", "vgg11_bn", "vgg19", "vgg19_bn"]
vit_models = ["vit_b_16", "vit_b_32"]


class MicroRobotModel(nn.Module):
    def __init__(self, num_outputs, in_channels=1):
        super().__init__()
        self.num_outputs = num_outputs
        self.in_channels = in_channels
        self.model = self._create_model()
        self._modify_first_layer()
        self._modify_last_layer()

    def _create_model(self):
        raise NotImplementedError

    def _modify_first_layer(self):
        raise NotImplementedError

    def _modify_last_layer(self):
        raise NotImplementedError

    def forward(self, x):
        return self.model(x)


class ResNetModel(MicroRobotModel):
    def __init__(self, model_name, num_outputs, in_channels=1):
        self.model_name = model_name
        super().__init__(num_outputs, in_channels)

    def _create_model(self):
        return getattr(models, self.model_name)()

    def _modify_first_layer(self):
        # ResNet expects conv1
        if self.in_channels != 3:
            old_layer = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                self.in_channels,
                old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

    def _modify_last_layer(self):
        # ResNet expects fc
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_outputs)


class ViTModel(MicroRobotModel):
    def __init__(self, model_name, num_outputs, in_channels=1):
        self.model_name = model_name
        super().__init__(num_outputs, in_channels)

    def _create_model(self):
        return getattr(models, self.model_name)()

    def _modify_first_layer(self):
        # ViT uses conv_proj for patch embeddings
        if self.in_channels != 3:
            old_layer = self.model.conv_proj
            self.model.conv_proj = nn.Conv2d(
                self.in_channels,
                old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

    def _modify_last_layer(self):
        # ViT heads is a Sequential block
        num_ftrs = self.model.heads[0].in_features
        self.model.heads = nn.Sequential(nn.Linear(num_ftrs, self.num_outputs))


class AlexNetModel(MicroRobotModel):
    def _create_model(self):
        return models.alexnet()

    def _modify_first_layer(self):
        # AlexNet features[0] is the first conv layer
        if self.in_channels != 3:
            old_layer = self.model.features[0]
            self.model.features[0] = nn.Conv2d(
                self.in_channels,
                old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

    def _modify_last_layer(self):
        # AlexNet classifier[6] is the last linear layer
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, self.num_outputs)


class VGGModel(MicroRobotModel):
    def __init__(self, model_name, num_outputs, in_channels=1):
        self.model_name = model_name
        super().__init__(num_outputs, in_channels)

    def _create_model(self):
        return getattr(models, self.model_name)()

    def _modify_first_layer(self):
        # VGG features[0] is the first conv layer
        if self.in_channels != 3:
            old_layer = self.model.features[0]
            self.model.features[0] = nn.Conv2d(
                self.in_channels,
                old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

    def _modify_last_layer(self):
        # VGG classifier[6] is the last linear layer
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, self.num_outputs)


def get_model(model_name: str, num_outputs: int, in_channels: int = 1):
    if model_name in resnet_models:
        return ResNetModel(model_name, num_outputs, in_channels)
    elif model_name in alexnet_models:
        return AlexNetModel(num_outputs, in_channels)
    elif model_name in vgg_models:
        return VGGModel(model_name, num_outputs, in_channels)
    elif model_name in vit_models:
        return ViTModel(model_name, num_outputs, in_channels)
    else:
        raise ValueError(f"Model {model_name} not supported.")
