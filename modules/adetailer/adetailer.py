import os
from typing import TYPE_CHECKING, Optional

from PIL import Image

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

    model_name = model_name or config.default_adetailer_model
    if model_name.startswith("mediapipe_"):
        # mediapipe models are builtin, no download required
        return mediapipe_predict(model_name, image, confidence=0.3)

    model_path = ensure_model(model_name)
    return ultralytics_predict(model_path, image, device="cpu")


def _apply_adetailer_single(image: Image.Image, model_name: str, tab_idx: int | None = None) -> Image.Image:
    """Run detection with a single model and blur detected regions."""
    result = detect(image, model_name)
    num_masks = len(result.masks)
    prefix = f"Tab {tab_idx}: " if tab_idx is not None else ""
    print(f"[ADetailer] {prefix}{num_masks} masks detected using {model_name}")
    if num_masks:
        from PIL import ImageFilter

        for idx, mask in enumerate(result.masks, 1):
            blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
            image.paste(blurred, mask=mask)
            print(f"[ADetailer] {prefix}Applied mask {idx}/{num_masks}")
    return image


def apply_adetailer_multi(image: Image.Image, params: Optional[dict] = None) -> Image.Image:
    """Apply ADetailer for all enabled tabs."""
    if not config.default_adetailer_enable:
        print("[ADetailer] disabled. skipping")
        return image
    try:
        any_enabled = False
        for i in range(1, TAB_COUNT + 1):
            if getattr(config, f"default_adetailer_tab{i}_enable", False):
                any_enabled = True
                _apply_adetailer_single(image, config.default_adetailer_model, tab_idx=i)
        if not any_enabled:
            print("[ADetailer] no tabs enabled. skipping")
    except Exception as e:  # pragma: no cover - best effort
        print(f"[ADetailer] failed: {e}")
    return image


# Backwards compatibility
def apply_adetailer(image: Image.Image) -> Image.Image:
    return apply_adetailer_multi(image)
