import os
from io import BytesIO
from typing import List
from PIL import Image
from modules import lora_utils
import shared
import gradio as gr

import json
import html
import datetime
from modules import util


preview_image = None


def pretty_bytes(num: int) -> str:
    """Return a human readable file size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"
try:
    import safetensors.torch as st
except Exception:
    st = None


get_lora_path = lora_utils.get_lora_path


read_metadata = lora_utils.read_metadata


read_user_metadata = lora_utils.read_user_metadata


write_user_metadata = lora_utils.write_user_metadata


list_loras = lora_utils.list_loras


find_preview = lora_utils.find_preview


build_tags = lora_utils.build_tags


def build_preview_html(path: str) -> str:
    """Return HTML snippet for preview image."""
    return (
        f"<div class='card standalone-card-preview'>"
        f"<img src=\"file={path}\" class=\"preview\"></div>"
    )


def build_filedata_table(path: str) -> str:
    """Return HTML table with basic file metadata."""
    try:
        stats = os.stat(path)
        size = pretty_bytes(stats.st_size)
        mtime = datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')
        filehash = util.sha256(path)
        table = "<table class='file-metadata'>"
        table += f"<tr><th>Filename:</th><td>{html.escape(os.path.basename(path))}</td></tr>"
        table += f"<tr><th>File size:</th><td>{size}</td></tr>"
        table += f"<tr><th>Hash:</th><td>{filehash}</td></tr>"
        table += f"<tr><th>Modified:</th><td>{mtime}</td></tr>"
        table += "</table>"
    except Exception:
        table = ""
    return table


def generate_cards():
    card_tpl = shared.html('extra-networks-card.html')
    copy_tpl = shared.html('extra-networks-copy-path-button.html')
    meta_tpl = shared.html('extra-networks-metadata-button.html')
    edit_tpl = shared.html('extra-networks-edit-item-button.html')
    files = list_loras()
    print('[LoRA UI] Detected files:', files)
    cards = []
    default_preview = os.path.join(
        os.path.dirname(__file__),
        "NewLoraSystem",
        "html",
        "card-no-preview.png",
    )
    for filepath in files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        preview_path = os.path.join("models", "loras_previews", f"{name}.preview.png")
        if os.path.exists(preview_path):
            preview = preview_path
        else:
            preview = find_preview(filepath)
            if not preview or not os.path.exists(preview):
                preview = default_preview
        preview_html = f'<img src="file={preview}" class="preview">'
        metadata = read_metadata(filepath)
        metadata_json = html.escape(json.dumps(metadata))
        user_meta = read_user_metadata(filepath)
        activation_text = user_meta.get('activation text', '')
        weight = float(user_meta.get('preferred weight', 1.0)) if user_meta.get('preferred weight') else 1.0
        prompt_text = f"<lora:{name}:{weight}>"
        if activation_text:
            prompt_text += f" {activation_text}"
        # Generate javascript for inserting the LoRA in the prompt.
        onclick_js = f"cardClicked('lora', \"{prompt_text}\", '' , false);"
        # Escape quotes but keep < and > intact inside the onclick attribute.
        onclick = html.escape(onclick_js, quote=True).replace("&lt;", "<").replace("&gt;", ">")
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
    cards_html = '\n'.join(cards)
    html_out = f"<div class='extra-network-pane'><div class='extra-network-cards'>{cards_html}</div></div>"
    print('[LoRA UI] HTML output:', cards_html[:200])
    return html_out


def load_editor(name):
    path = get_lora_path(name)
    if not path:
        return [name, '', '', 'Unknown', [], '', '', 1.0, '', '', '']

    metadata = read_metadata(path)
    user_meta = read_user_metadata(path)

    desc = user_meta.get('description', '')
    activation = user_meta.get('activation text', '')
    weight = float(user_meta.get('preferred weight', 1.0)) if user_meta.get('preferred weight') else 1.0
    negative = user_meta.get('negative text', '')
    notes = user_meta.get('notes', '')
    sd_version = user_meta.get('sd version', 'Unknown')

    tags_list = build_tags(metadata)
    tag_pairs = [(t, str(c)) for t, c in tags_list[:20]]
    tags_text = ', '.join([t for t, _ in tags_list])

    filedata = build_filedata_table(path)
    default_preview = os.path.join(
        os.path.dirname(__file__),
        "NewLoraSystem",
        "html",
        "card-no-preview.png",
    )
    preview_path = os.path.join("models", "loras_previews", f"{name}.preview.png")
    if os.path.exists(preview_path):
        preview = preview_path
    else:
        preview = find_preview(path)
        if not preview or not os.path.exists(preview):
            preview = default_preview

    preview_html = (
        f"<div class='card standalone-card-preview'>"
        f"<img src=\"file={preview}\" class=\"preview\"></div>"
    )

    return [
        name,
        desc,
        filedata,
        sd_version,
        tag_pairs,
        tags_text,
        activation,
        weight,
        negative,
        notes,
        preview_html,
    ]


def save_metadata(name, description, activation, weight, negative, notes, sd_version):
    path = get_lora_path(name)
    if not path:
        return ''
    data = read_user_metadata(path)
    data['description'] = description
    data['activation text'] = activation
    data['preferred weight'] = weight
    data['negative text'] = negative
    data['notes'] = notes
    data['sd version'] = sd_version
    write_user_metadata(path, data)
    return ''


def save_preview(name, *_):
    """Save the most recent Fooocus result as the preview for ``name``."""
    import modules.images_output as images_output

    global preview_image

    path = get_lora_path(name)
    if not path:
        return ''

    img_data = images_output.get_last_result_image()
    if img_data is None:
        raise ValueError("No se encontró imagen reciente para usar como preview.")

    if not isinstance(img_data, Image.Image):
        try:
            img_data = Image.fromarray(img_data)
        except Exception as e:  # pragma: no cover - image conversion failures
            raise ValueError(f"Error al convertir la imagen: {e}")

    preview_dir = os.path.join('models', 'loras_previews')
    os.makedirs(preview_dir, exist_ok=True)

    preview_path = os.path.join(preview_dir, f"{name}.preview.png")
    img_data.save(preview_path)

    return {preview_image: gr.HTML.update(value=build_preview_html(preview_path))}


def create_editor_ui(tabname: str, gallery, prompt):
    with gr.Box(visible=False, elem_id=f"{tabname}_lora_edit_user_metadata", elem_classes="edit-user-metadata") as box:
        name_in = gr.Textbox(visible=False, elem_id=f"{tabname}_lora_edit_user_metadata_name")
        button_edit = gr.Button("Edit user metadata", visible=False, elem_id=f"{tabname}_lora_edit_user_metadata_button")
        global preview_image
        with gr.Row(equal_height=True):
            with gr.Column(scale=7, min_width=400):
                title = gr.HTML()
                desc = gr.Textbox(label="Description", lines=4)
                filedata_html = gr.HTML()
                sd_version = gr.Dropdown(['SD1', 'SD2', 'SDXL', 'Unknown'], value='Unknown', label='Stable Diffusion version')
            with gr.Column(scale=3, min_width=200):
                preview_image = gr.HTML(
                    elem_id="lora_preview_image"
                )

    taginfo = gr.HighlightedText(label='Training dataset tags \U0001F4D0')
    tags_text = gr.Textbox(visible=False)
    add_tags = gr.Button('Add tags to prompt')
    activation = gr.Textbox(label="Activation text")
    weight = gr.Slider(label="Preferred weight", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
    negative = gr.Textbox(label="Negative prompt")
    notes = gr.TextArea(label="Notes", lines=4)
    status = gr.HTML()

    with gr.Row():
        cancel = gr.Button('Cancel')
        replace_preview = gr.Button('Replace preview', variant='primary')
        save = gr.Button('Save', variant='primary')

        cancel.click(fn=None, _js="closePopup")

        add_tags.click(
            fn=None,
            _js=f"function(){{addTagsToPrompt(gradioApp().querySelector('#{tags_text.elem_id} textarea').value, '{tabname}')}}",
            inputs=[],
            outputs=[],
            show_progress=False,
        )

        button_edit.click(
            fn=load_editor,
            inputs=[name_in],
            outputs=[title, desc, filedata_html, sd_version, taginfo, tags_text, activation, weight, negative, notes, preview_image],
        ).then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[box])

        save.click(fn=save_metadata, inputs=[name_in, desc, activation, weight, negative, notes, sd_version], outputs=status).then(fn=None, _js='refreshLoraCards')

        replace_preview.click(
            fn=save_preview,
            inputs=[name_in],
            outputs=preview_image
        ).then(fn=None, _js='refreshLoraCards')

    return name_in, button_edit


def setup_ui(tabname: str, gallery, prompt):
    return create_editor_ui(tabname, gallery, prompt)
