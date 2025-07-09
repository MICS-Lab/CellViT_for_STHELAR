import torch
from .lora import LoRA
from .plora import PLoRA
from .bottleneck import BottleNeck
from .adaptformer import AdaptFormer

def insert_plora(model, rank, alpha):
    """
        Insert PLoRA into a model (expects the timm ViT architecture, but can easily be adapted to others).
        arguments:
            model [nn.Module]: model to insert PLoRA into
            rank [int]: rank of PLoRA
            alpha [float]: alpha parameter scaling the adaptation
        returns:
            None
    """
    for encoder_block in model.encoder.blocks:
        encoder_block.attn.qkv = PLoRA(linear_layer=encoder_block.attn.qkv,
                                       rank=rank,
                                       alpha=alpha)

def insert_lora(model, rank, alpha):
    """
        Insert LoRA into a model (expects the timm ViT architecture, but can easily be adapted to others).
        arguments:
            model [nn.Module]: model to insert LoRA into
            rank [int]: rank of LoRA
            alpha [float]: alpha parameter scaling the adaptation
        returns:
            None
    """
    for encoder_block in model.encoder.blocks:
        encoder_block.attn.qkv = LoRA(linear_layer=encoder_block.attn.qkv,
                                      rank=rank,
                                      alpha=alpha)

def insert_adaptformer(model, activation, reduction):
        """
            Insert AdaptFormer into a model (expects the timm ViT architecture, but can easily be adapted to others).
            arguments:
                model [nn.Module]: model to insert AdaptFormer into
                activation [str]: activation function to use in the adapter
                reduction [int]: reduction factor of the adapter
            returns:
                None
        """
        for encoder_block in model.encoder.blocks:
            encoder_block.mlp = AdaptFormer(encoder_block.norm2, encoder_block.mlp, activation, reduction)
            encoder_block.norm2 = torch.nn.Identity()

def insert_bottleneck(model, activation, reduction):
    """
        Insert a BottleNeck adapter into a model (expects the timm ViT architecture, but can easily be adapted to others).
        arguments:
            model [nn.Module]: model to insert the BottleNeck into
            activation [str]: activation function to use in the adapter
            reduction [int]: reduction factor of the adapter
        returns:
            None
    """
    for encoder_block in model.encoder.blocks:
        encoder_block.attn = torch.nn.Sequential(encoder_block.attn,
                                                 BottleNeck(encoder_block.attn.proj.out_features,
                                                            activation,
                                                            reduction))
        encoder_block.mlp = torch.nn.Sequential(encoder_block.mlp,
                                                BottleNeck(encoder_block.mlp.lin2.out_features,
                                                           activation,
                                                           reduction))

def freeze_model(model, adapter_type):
    """
        Freeze the whole model except for the plora layers.
        arguments:
            model [nn.Module]: model to freeze
            adapter_type [str]: type of adapter used
        returns:
            None
    """
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        if adapter_type == 'bottleneck' and 'norm' in name and 'block' in name:
            param.requires_grad = True