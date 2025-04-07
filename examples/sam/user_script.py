from transformers import SamModel

def encoder_loader(model_path: str):
    model = SamModel.from_pretrained(model_path)
    return model.vision_encoder
