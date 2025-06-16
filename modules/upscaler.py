from collections import OrderedDict
import gc
import numpy as np

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from ldm_patched.utils import path_utils
from modules.config import downloading_upscale_model

opImageUpscaleWithModel = ImageUpscaleWithModel()

_cached_model = None
_cached_model_name = None


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def perform_upscale(img, upscaler_name=None):
    """Upscale ``img`` (``HWC`` or ``NHWC``) using the given ESRGAN model name."""
    global _cached_model, _cached_model_name

    print(f'Upscaling image with shape {str(img.shape)} ...')

    if upscaler_name is None:
        upscaler_name = 'default'

    if _cached_model_name != upscaler_name:
        if _cached_model is not None:
            del _cached_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_path = path_utils.get_full_path('upscale_models', upscaler_name)
        if model_path is None:
            model_path = downloading_upscale_model()
        sd = torch.load(model_path, map_location='cpu', weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        _cached_model = ESRGAN(sdo)
        _cached_model.eval()
        _cached_model_name = upscaler_name

    device = _get_device()
    _cached_model.to(device)

    # support single image (HWC) or batch (NHWC)
    if img.ndim == 3:
        tensor = core.numpy_to_pytorch(img).to(device)
    else:
        imgs = [(im.astype(np.float32) / 255.0) for im in img]
        tensor = torch.from_numpy(np.stack(imgs)).float().to(device)

    tensor = opImageUpscaleWithModel.upscale(_cached_model, tensor)[0]
    result_list = core.pytorch_to_numpy(tensor.cpu())

    _cached_model.to('cpu')

    if img.ndim == 3:
        return result_list[0]
    return np.stack(result_list)
