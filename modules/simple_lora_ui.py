import os
from io import BytesIO
from typing import List
from PIL import Image
from modules import config, util
import shared
import gradio as gr
import json
import html
try:
    import safetensors.torch as st
except Exception:
    st = None

LORA_EXTENSIONS = {'.safetensors', '.ckpt', '.pt'}


def get_lora_path(name: str) -> str | None:
    for folder in config.paths_loras:
        for ext in LORA_EXTENSIONS:
            path = os.path.join(folder, name + ext)
            if os.path.exists(path):
                return path
    return None


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


def read_user_metadata(path: str) -> dict:
    json_path = os.path.splitext(path)[0] + '.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def write_user_metadata(path: str, data: dict):
    json_path = os.path.splitext(path)[0] + '.json'
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def list_loras():
    files = []
    for folder in config.paths_loras:
        if not os.path.isdir(folder):
            continue
        for root, _dirs, filenames in os.walk(folder):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in LORA_EXTENSIONS:
                    files.append(os.path.join(root, name))
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


def build_tags(metadata: dict) -> List[str]:
    tags = {}
    freq = metadata.get('ss_tag_frequency', {})
    if hasattr(freq, 'items'):
        for _, d in freq.items():
            for tag, count in d.items():
                tag = tag.strip()
                tags[tag] = tags.get(tag, 0) + int(count)
    ordered = sorted(tags.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in ordered]


def generate_cards():
    card_tpl = shared.html('extra-networks-card.html')
    copy_tpl = shared.html('extra-networks-copy-path-button.html')
    meta_tpl = shared.html('extra-networks-metadata-button.html')
    edit_tpl = shared.html('extra-networks-edit-item-button.html')
    files = list_loras()
    print('[LoRA UI] Detected files:', files)
    cards = []
    for filepath in files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        preview = find_preview(filepath)
        if preview:
            preview_html = f'<img src="file={preview}" class="preview">'
        else:
            preview_html = ''
        metadata = read_metadata(filepath)
        metadata_json = html.escape(json.dumps(metadata))
        user_meta = read_user_metadata(filepath)
        activation_text = user_meta.get('activation text', '')
        weight = float(user_meta.get('preferred weight', 1.0)) if user_meta.get('preferred weight') else 1.0
        prompt_text = f"<lora:{name}:{weight}>"
        if activation_text:
            prompt_text += f" {activation_text}"
        prompt = f"\"{prompt_text}\""
        onclick = html.escape(f"cardClicked('advanced', {prompt}, '' , false);")
        args = {
            'style': '',
            'card_clicked': onclick,
            'name': name,
            'filename': filepath,
            'sort_keys': '',
            'background_image': preview_html,
            'copy_path_button': copy_tpl.format(filename=filepath),
            'metadata_button': meta_tpl.format(extra_networks_tabname='lora', metadata_json=metadata_json),
            'edit_button': edit_tpl.format(tabname='advanced', extra_networks_tabname='lora'),
            'description': '',
            'search_terms': '',
        }
        cards.append(card_tpl.format(**args))
    html_out = '\n'.join(cards)
    print('[LoRA UI] HTML output:', html_out[:200])
    return html_out


def load_editor(name):
    path = get_lora_path(name)
    if not path:
        return [name, '', '', 1.0, '', 'Unknown', '', '']
    metadata = read_metadata(path)
    user_meta = read_user_metadata(path)
    desc = user_meta.get('description', '')
    activation = user_meta.get('activation text', '')
    weight = float(user_meta.get('preferred weight', 1.0)) if user_meta.get('preferred weight') else 1.0
    notes = user_meta.get('notes', '')
    sd_version = user_meta.get('sd version', 'Unknown')
    tags = ', '.join(build_tags(metadata)[:20])
    preview = find_preview(path)
    if preview:
        preview_html = f'<img src="file={preview}" class="preview">'
    else:
        preview_html = ''
    return [name, desc, activation, weight, notes, sd_version, tags, preview_html]


def save_metadata(name, description, activation, weight, notes, sd_version):
    path = get_lora_path(name)
    if not path:
        return ''
    data = read_user_metadata(path)
    data['description'] = description
    data['activation text'] = activation
    data['preferred weight'] = weight
    data['notes'] = notes
    data['sd version'] = sd_version
    write_user_metadata(path, data)
    return ''


def save_preview(name, gallery: List, index: int):
    path = get_lora_path(name)
    if not path:
        return ''
    if not gallery:
        return ''
    index = int(index)
    index = max(0, min(index, len(gallery) - 1))
    img_data = gallery[index]
    if isinstance(img_data, str):
        if img_data.startswith('data:'):
            import base64
            img_data = Image.open(BytesIO(base64.b64decode(img_data.split(',')[1])))
        else:
            img_data = Image.open(img_data)
    else:
        img_data = Image.fromarray(img_data) if not isinstance(img_data, Image.Image) else img_data
    preview_path = os.path.splitext(path)[0] + '.preview.png'
    img_data.save(preview_path)
    return ''


def create_editor_ui(tabname: str, gallery, prompt):
    with gr.Box(visible=False, elem_id=f"{tabname}_lora_edit_user_metadata", elem_classes="edit-user-metadata"):
        name_in = gr.Textbox(visible=False, elem_id=f"{tabname}_lora_edit_user_metadata_name")
        button_edit = gr.Button("Edit user metadata", visible=False, elem_id=f"{tabname}_lora_edit_user_metadata_button")
        title = gr.HTML()
        desc = gr.Textbox(label="Description", lines=4)
        activation = gr.Textbox(label="Activation text")
        weight = gr.Slider(label="Preferred weight", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
        notes = gr.TextArea(label="Notes", lines=4)
        sd_version = gr.Dropdown(['SD1', 'SD2', 'SDXL', 'Unknown'], value='Unknown', label='Stable Diffusion version')
        tags = gr.Textbox(label='Tags', interactive=False)
        add_tags = gr.Button('Add tags to prompt')
        preview_html = gr.HTML()
        status = gr.HTML()
        with gr.Row():
            cancel = gr.Button('Cancel')
            replace_preview = gr.Button('Replace preview', variant='primary')
            save = gr.Button('Save', variant='primary')

        cancel.click(fn=None, _js="closePopup")

        def add_tags_fn(t, p):
            if not t:
                return p
            if p:
                p += ', '
            p += t
            return p

        add_tags.click(fn=add_tags_fn, inputs=[tags, prompt], outputs=prompt, show_progress=False)

        button_edit.click(fn=load_editor, inputs=[name_in], outputs=[title, desc, activation, weight, notes, sd_version, tags, preview_html])

        save.click(fn=save_metadata, inputs=[name_in, desc, activation, weight, notes, sd_version], outputs=status).then(fn=None, _js='refreshLoraCards')

        replace_preview.click(
            fn=lambda name, g, idx: save_preview(name, g, idx),
            _js="function(a,b){return [a,b,selected_gallery_index()]}",
            inputs=[name_in, gallery], outputs=status
        ).then(fn=None, _js='refreshLoraCards')

    return name_in, button_edit


def setup_ui(tabname: str, gallery, prompt):
    return create_editor_ui(tabname, gallery, prompt)
