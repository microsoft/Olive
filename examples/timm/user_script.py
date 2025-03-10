def load_timm(model_name: str):
    import timm

    model = timm.create_model(model_name, pretrained=True)
    return model.eval()
