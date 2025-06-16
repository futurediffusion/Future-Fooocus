import os
from modules import config, util, shared
import json
import html
try:
    import safetensors.torch as st
except Exception:
    st = None

LORA_EXTENSIONS = {'.safetensors', '.ckpt', '.pt'}


def read_metadata(path):
    metadata = {}
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
    card_tpl = shared.html('extra-networks-card.html')
    copy_tpl = shared.html('extra-networks-copy-path-button.html')
    meta_tpl = shared.html('extra-networks-metadata-button.html')
    edit_tpl = shared.html('extra-networks-edit-item-button.html')
    cards = []
    for filepath in list_loras():
        name = os.path.splitext(os.path.basename(filepath))[0]
        preview = find_preview(filepath)
        if preview:
            preview_html = f'<img src="file={preview}" class="preview">'
        else:
            preview_html = ''
        prompt = f"\"<lora:{name}:1>\""
        onclick = f"cardClicked('advanced', {prompt}, '' , false);"
        metadata = read_metadata(filepath)
        metadata_json = html.escape(json.dumps(metadata))
        args = {
            'style': '',
            'card_clicked': onclick,
            'name': name,
            'sort_keys': '',
            'background_image': preview_html,
            'copy_path_button': copy_tpl.format(filename=filepath),
            'metadata_button': meta_tpl.format(extra_networks_tabname='lora', metadata_json=metadata_json),
            'edit_button': edit_tpl.format(tabname='advanced', extra_networks_tabname='lora'),
            'description': '',
            'search_terms': '',
        }
        cards.append(card_tpl.format(**args))
    return '\n'.join(cards)
