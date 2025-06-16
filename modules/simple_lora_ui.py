import os
import json
from typing import Dict

from modules import config, util, shared
from safetensors import safe_open

_lora_data: Dict[str, dict] = {}

LORA_EXTENSIONS = {'.safetensors', '.ckpt', '.pt'}


def list_loras():
    files = []
    for folder in config.paths_loras:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            ext = os.path.splitext(name)[1].lower()
            if ext in LORA_EXTENSIONS:
                files.append(os.path.join(folder, name))
    return sorted(files)


def load_metadata(path: str) -> dict:
    if not path.lower().endswith('.safetensors'):
        return {}
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.metadata()
    except Exception:
        return {}


def scan_loras() -> None:
    """Populate internal cache with lora info."""
    _lora_data.clear()
    for filepath in list_loras():
        name = os.path.splitext(os.path.basename(filepath))[0]
        _lora_data[name] = {
            'path': filepath,
            'preview': find_preview(filepath),
            'metadata': load_metadata(filepath),
        }


def find_preview(path):
    base, _ = os.path.splitext(path)
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        cand = base + ext
        if os.path.exists(cand):
            return cand
        cand = base + '.preview' + ext
        if os.path.exists(cand):
            return cand
    return None


def generate_cards():
    scan_loras()
    card_tpl = shared.html('extra-networks-card.html')
    copy_tpl = shared.html('extra-networks-copy-path-button.html')
    meta_tpl = shared.html('extra-networks-metadata-button.html')
    edit_tpl = shared.html('extra-networks-edit-item-button.html')
    cards = []
    for name, info in _lora_data.items():
        filepath = info['path']
        preview = info['preview']
        if preview:
            preview_html = f'<img src="file={preview}" class="preview">'
        else:
            preview_html = ''
        prompt = f"\"<lora:{name}:1>\""
        onclick = f"cardClicked('advanced', {prompt}, '' , false);"
        args = {
            'style': '',
            'card_clicked': onclick,
            'name': name,
            'sort_keys': '',
            'background_image': preview_html,
            'copy_path_button': copy_tpl.format(filename=filepath),
            'metadata_button': meta_tpl.format(extra_networks_tabname='lora'),
            'edit_button': edit_tpl.format(tabname='advanced', extra_networks_tabname='lora'),
            'description': '',
            'search_terms': '',
        }
        cards.append(card_tpl.format(**args))
    return '\n'.join(cards)


def get_metadata(name: str) -> dict | None:
    if not _lora_data:
        scan_loras()
    info = _lora_data.get(name)
    return info.get('metadata') if info else None
