import os
from typing import TYPE_CHECKING, Optional

from PIL import Image, ImageFilter
import numpy as np

# enable saving intermediate crops if environment variable is set
ADETAILER_DEBUG = os.getenv("ADETAILER_DEBUG", "0").lower() in {"1", "true", "yes"}

from .vendor_adetailer.common import ensure_pil_image

from modules import config
from modules.model_loader import load_file_from_url

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .vendor_adetailer import PredictOutput

MODEL_URLS = {
    "face_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
    "face_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt",
    "hand_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt",
    "person_yolov8n-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt",
    "yolov8x-worldv2.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/yolov8x-worldv2.pt",
}
TAB_COUNT = 4


def ensure_model(model_name: str, url: Optional[str] = None) -> str:
    os.makedirs(config.path_adetailer_detection, exist_ok=True)
    model_path = os.path.join(config.path_adetailer_detection, model_name)
    if not os.path.exists(model_path):
        target_url = url or MODEL_URLS.get(model_name)
        if target_url:
            print(f"[ADetailer] Downloading model {model_name} ...")
            try:
                load_file_from_url(
                    url=target_url,
                    model_dir=config.path_adetailer_detection,
                    file_name=model_name,
                )
            except Exception as e:  # pragma: no cover - best effort
                print(
                    f"[ADetailer] failed to download {model_name}: {e}. "
                    "Please download manually and place under models/detection/adetailer."
                )
    return model_path


def detect(image: Image.Image, model_name: Optional[str] = None) -> "PredictOutput":
    from .vendor_adetailer import mediapipe_predict, ultralytics_predict

    image = ensure_pil_image(image, "RGB")
    model_name = model_name or config.default_adetailer_model
    if model_name.startswith("mediapipe_"):
        # mediapipe models are builtin, no download required
        return mediapipe_predict(model_name, image, confidence=0.3)

    model_path = ensure_model(model_name)
    return ultralytics_predict(model_path, image, device="cpu")


def _refine_mask_region(image: Image.Image, mask: Image.Image, idx: int = 0) -> None:
    """Refine a masked region using upscale, filters and feathered blending."""
    bbox = mask.getbbox()
    if not bbox:
        return

    x1, y1, x2, y2 = bbox
    pad = int(0.1 * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)

    region = image.crop((x1, y1, x2, y2))
    mask_crop = mask.crop((x1, y1, x2, y2))

    if ADETAILER_DEBUG:
        os.makedirs("debug", exist_ok=True)
        region.save(f"debug/mask_{idx}_before.png")

    scale = 2
    up_size = (region.width * scale, region.height * scale)
    upscaled = region.resize(up_size, Image.LANCZOS)
    upscaled = upscaled.filter(ImageFilter.DETAIL)
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    refined = upscaled.resize(region.size, Image.LANCZOS)

    if ADETAILER_DEBUG:
        refined.save(f"debug/mask_{idx}_after.png")

    feather_radius = max(1, int(0.05 * max(region.size)))
    mask_blur = mask_crop.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    image.paste(refined, (x1, y1, x2, y2), mask=mask_blur)


def _apply_adetailer_single(image: Image.Image, model_name: str, tab_idx: int | None = None) -> Image.Image:
    """Run detection with a single model and refine detected regions."""
    result = detect(image, model_name)
    num_masks = len(result.masks)
    prefix = f"Tab {tab_idx}: " if tab_idx is not None else ""
    print(f"[Adetailer] {prefix}{num_masks} masks detected using {model_name}")
    if num_masks:
        for idx, mask in enumerate(result.masks, 1):
            _refine_mask_region(image, mask, idx)
        print(f"[Adetailer] Applied {num_masks} masks on tab {tab_idx}")
    return image


def apply_adetailer_multi(image: Image.Image | np.ndarray, params: Optional[dict] = None) -> Image.Image | np.ndarray:
    """Apply ADetailer for all enabled tabs."""
    if not config.default_adetailer_enable:
        print("[ADetailer] disabled. skipping")
        return image

    return_np = not isinstance(image, Image.Image)
    pil_img = ensure_pil_image(image)

    try:
        enabled_tabs = [
            str(i)
            for i in range(1, TAB_COUNT + 1)
            if getattr(config, f"default_adetailer_tab{i}_enable", False)
        ]
        if not enabled_tabs:
            print("[Adetailer] no tabs enabled. skipping")
            return image

        print(f"[Adetailer] Enabled tabs: {', '.join(enabled_tabs)}")
        print(f"[Adetailer] Using model: {config.default_adetailer_model}")

        for i in map(int, enabled_tabs):
            pil_img = _apply_adetailer_single(pil_img, config.default_adetailer_model, tab_idx=i)
    except Exception as e:  # pragma: no cover - best effort
        print(f"[ADetailer] failed: {e}")
    if return_np:
        return np.array(pil_img)
    return pil_img


# Backwards compatibility
def apply_adetailer(image: Image.Image | np.ndarray) -> Image.Image | np.ndarray:
    return apply_adetailer_multi(image)
