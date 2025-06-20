import os
from typing import Optional, TYPE_CHECKING
from PIL import Image

from modules.model_loader import load_file_from_url
import modules.config as config

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


def ensure_model(model_name: str, url: Optional[str] = None) -> str:
    os.makedirs(config.path_adetailer_detection, exist_ok=True)
    model_path = os.path.join(config.path_adetailer_detection, model_name)
    if not os.path.exists(model_path):
        target_url = url or MODEL_URLS.get(model_name)
        if target_url:
            print(f"[ADetailer] Downloading model {model_name} ...")
            load_file_from_url(url=target_url, model_dir=config.path_adetailer_detection, file_name=model_name)
    return model_path


def detect(image: Image.Image, model_name: Optional[str] = None) -> "PredictOutput":
    from .vendor_adetailer import ultralytics_predict, mediapipe_predict

    model_name = model_name or config.default_adetailer_model
    if model_name.startswith("mediapipe_"):
        # mediapipe models are builtin, no download required
        return mediapipe_predict(model_name, image, confidence=0.3)

    model_path = ensure_model(model_name)
    return ultralytics_predict(model_path, image, device="cpu")


def apply_adetailer(image: Image.Image) -> Image.Image:
    if not config.default_adetailer_enable:
        return image
    try:
        result = detect(image)
        num_masks = len(result.masks)
        print(
            f"[ADetailer] {num_masks} masks detected using {config.default_adetailer_model}"
        )

        if num_masks:
            from PIL import ImageFilter

            for idx, mask in enumerate(result.masks, 1):
                # simple blur over detected region as placeholder for real inpainting
                blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
                image.paste(blurred, mask=mask)
                print(f"[ADetailer] Applied mask {idx}/{num_masks}")
    except Exception as e:  # pragma: no cover - best effort
        print(f"[ADetailer] failed: {e}")
    return image
