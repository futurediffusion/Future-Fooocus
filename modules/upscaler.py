"""Utility helpers for ESRGAN based upscaling."""

from collections import OrderedDict

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.model_loading import load_state_dict
from ldm_patched.utils import path_utils
from modules.config import downloading_upscale_model

opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None
model_name = None


def load_upscaler_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load an ESRGAN model by name onto the given device."""
    model_path = path_utils.get_full_path("upscale_models", model_name)
    sd = torch.load(model_path, weights_only=True)
    # Replace older naming convention
    sdo = OrderedDict()
    for k, v in sd.items():
        sdo[k.replace('residual_block_', 'RDB')] = v
    del sd
    model = load_state_dict(sdo)
    model.to(device)
    model.eval()
    return model


def upscale_with_model(model: torch.nn.Module, img_np, device: torch.device):
    """Upscale ``img_np`` (HWC numpy) using the provided ESRGAN model."""
    img = core.numpy_to_pytorch(img_np).to(device)
    out = opImageUpscaleWithModel.upscale(model, img)[0]
    out = core.pytorch_to_numpy(out)[0]
    return out


def perform_upscale(img):
    global model, model_name

    print(f'Upscaling image with shape {str(img.shape)} ...')

    if model is None:
        model_filename = downloading_upscale_model()
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = load_state_dict(sdo)
        model.cpu()
        model.eval()
        model_name = model_filename

    img = upscale_with_model(model, img, torch.device('cpu'))
    
    return img
