from importlib import import_module

torch_vision = "torchvision.models"
MODEL_MAP = {
    "resnet18": torch_vision,
    "resnet34": torch_vision,
    "resnet50": torch_vision,
}


def get_model(model_name: str, *arg, **kwg):
    if model_name not in MODEL_MAP:
        return None

    module = MODEL_MAP.get(model_name)

    try:
        mod = import_module(module)
        cls = getattr(mod, model_name)
        return cls(*arg, **kwg)
    except (ImportError, AttributeError):
        return None
