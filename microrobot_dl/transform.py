import torchvision.transforms as T
import torch


class TranslateTransform:
    def __init__(self, translate: tuple[float, float] = (0.1, 0.1)):
        self.translate = translate
        self.translate_transform = T.RandomAffine(
            degrees=0,
            translate=self.translate,
        )

    def __call__(self, img):
        return self.translate_transform(img)


class ZoomTransform:
    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float],
        ratio: tuple[float, float] = (1.0, 1.0),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.zoom_transform = T.RandomResizedCrop(
            size=self.size,
            scale=self.scale,
            ratio=self.ratio,
        )

    def __call__(self, img):
        return self.zoom_transform(img)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


train_transform = T.Compose(
    [
        T.Resize((240, 240)),
        TranslateTransform((0.05, 0.05)),
        ZoomTransform(size=(224, 224), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        AddGaussianNoise(0.0, 0.02),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
)
