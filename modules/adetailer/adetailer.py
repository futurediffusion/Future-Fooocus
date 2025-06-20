import os
from typing import Optional, TYPE_CHECKING, Iterable
from PIL import Image, ImageFilter

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
    "yolov8x-worldv2.pt": "https://huggingface.co/Bingsu/yolo-world-mirror/resolve/main/yolov8x-worldv2.pt",
    "mediapipe_face_full": None,
    "mediapipe_face_short": None,
    "mediapipe_face_mesh": None,
    "mediapipe_face_mesh_eyes_only": None,
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


def detect(image: Image.Image, model_name: Optional[str] = None, confidence: float = 0.3, classes: str = "") -> "PredictOutput":
    from .vendor_adetailer import ultralytics_predict, mediapipe_predict
    model_name = model_name or config.default_adetailer_model
    if model_name.startswith("mediapipe"):
        return mediapipe_predict(model_name, image, confidence=confidence)
    model_path = ensure_model(model_name)
    return ultralytics_predict(model_path, image, confidence=confidence, device="cpu", classes=classes)


def apply_adetailer(image: Image.Image, models: Iterable[str] | None = None) -> Image.Image:
    """Apply ADetailer to ``image`` using the given models.

    This implementation is simplified and applies a detail filter to detected regions.
    """

    if not config.default_adetailer_enable:
        return image

    models = list(models or [config.default_adetailer_model])
    result_img = image.copy()

    for model in models:
        try:
            result = detect(result_img, model)
            print(f"[ADetailer] {len(result.masks)} masks detected using {model}")
            for mask in getattr(result, "masks", []):
                filtered = result_img.filter(ImageFilter.DETAIL)
                result_img.paste(filtered, mask=mask)
        except Exception as e:  # pragma: no cover - best effort
            print(f"[ADetailer] failed on {model}: {e}")

    return result_img
