import timm


def build_model(model_name: str, num_classes: int, pretrained: bool = True, cache_dir: str = ""):
    kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return timm.create_model(model_name, **kwargs)
