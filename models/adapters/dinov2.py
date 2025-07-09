import torch

def get_dinov2(model_type='dinov2_vitb14'):
    model = torch.hub.load('facebookresearch/dinov2', model_type)
    return model