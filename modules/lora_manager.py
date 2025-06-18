# Backend utilities for LoRA management
# Provides helpers to list LoRA files, load metadata and previews,
# and read/write user metadata JSON files.

import os
import json
from typing import List, Dict, Optional

from PIL import Image

from modules import config

try:
    import safetensors.torch as st
except Exception:  # pragma: no cover - optional dependency
    st = None

LORA_EXTENSIONS = {'.safetensors', '.ckpt', '.pt'}


def get_lora_path(name: str) -> Optional[str]:
    """Return the first existing path for a LoRA model matching ``name``."""
    for folder in config.paths_loras:
        for ext in LORA_EXTENSIONS:
            path = os.path.join(folder, name + ext)
            if os.path.exists(path):
                return path
    return None


def read_metadata(path: str) -> Dict:
    """Read metadata from a LoRA file and its companion JSON."""
    metadata: Dict = {}
    if path.lower().endswith('.safetensors') and st is not None:
        try:
            with st.safe_open(path, framework="pt") as f:
                metadata.update(f.metadata())
        except Exception:
            pass

    json_path = os.path.splitext(path)[0] + '.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf8') as j:
                metadata.update(json.load(j))
        except Exception:
            pass

    return metadata


def read_user_metadata(path: str) -> Dict:
    """Load user metadata stored next to the LoRA file."""
    json_path = os.path.splitext(path)[0] + '.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def write_user_metadata(path: str, data: Dict) -> None:
    """Write user metadata to ``<lora>.json``."""
    json_path = os.path.splitext(path)[0] + '.json'
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def list_loras() -> List[str]:
    """Return a sorted list of LoRA filepaths discovered in configured directories."""
    files: List[str] = []
    for folder in config.paths_loras:
        if not os.path.isdir(folder):
            continue
        for root, _dirs, filenames in os.walk(folder):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in LORA_EXTENSIONS:
                    files.append(os.path.join(root, name))
    return sorted(files)


def find_preview(path: str) -> Optional[str]:
    """Look for an image preview next to the given LoRA file."""
    base, _ = os.path.splitext(path)
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        cand = base + ext
        if os.path.exists(cand):
            return cand
        cand = base + '.preview' + ext
        if os.path.exists(cand):
            return cand
    return None


def save_preview_image(path: str, image, fmt: str = 'PNG') -> None:
    """Save ``image`` as a preview next to ``path``."""
    preview_path = os.path.splitext(path)[0] + '.preview.png'
    image.save(preview_path, format=fmt)


def build_tags(metadata: Dict) -> List[tuple[str, int]]:
    """Return list of ``(tag, count)`` sorted by frequency."""
    tags: Dict[str, int] = {}
    freq = metadata.get('ss_tag_frequency', {})
    if hasattr(freq, 'items'):
        for _key, data in freq.items():
            for tag, count in data.items():
                tag = tag.strip()
                tags[tag] = tags.get(tag, 0) + int(count)

    ordered = sorted(tags.items(), key=lambda x: x[1], reverse=True)
    return ordered

