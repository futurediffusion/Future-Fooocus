# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import json
import gradio as gr
import args_manager
import modules.config

from modules.localization import localization_js


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def javascript_html():
    script_js_path = webpath('javascript/script.js')
    context_menus_js_path = webpath('javascript/contextMenus.js')
    localization_js_path = webpath('javascript/localization.js')
    zoom_js_path = webpath('javascript/zoom.js')
    edit_attention_js_path = webpath('javascript/edit-attention.js')
    viewer_js_path = webpath('javascript/viewer.js')
    image_viewer_js_path = webpath('javascript/imageviewer.js')
    zoom_image_js_path = webpath('javascript/zoomimage.js')
    tag_autocomplete_js_path = webpath('javascript/tag_autocomplete.js')
    tag_dir = os.path.join(script_path, 'a1111-sd-webui-tagcomplete', 'tags')
    tag_files = [os.path.join('a1111-sd-webui-tagcomplete', 'tags', f) for f in os.listdir(tag_dir) if f.endswith('.csv')]
    chant_files = [os.path.join('a1111-sd-webui-tagcomplete', 'tags', f) for f in os.listdir(tag_dir) if f.endswith('-chants.json')]
    tag_files_json = json.dumps(tag_files)
    chant_files_json = json.dumps(chant_files)
    tac_cfg = json.dumps({
        'enabled': modules.config.tac_active,
        'tagFile': os.path.join('a1111-sd-webui-tagcomplete', 'tags', modules.config.tac_tag_file),
        'chantFile': os.path.join('a1111-sd-webui-tagcomplete', 'tags', modules.config.tac_chant_file),
        'maxResults': modules.config.tac_max_results,
        'appendComma': modules.config.tac_append_comma,
        'appendSpace': modules.config.tac_append_space,
        'replaceUnderscores': modules.config.tac_replace_underscores,
        'escapeParentheses': modules.config.tac_escape_parentheses,
    })
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    head += f'<script type="text/javascript" src="{script_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{context_menus_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{localization_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{zoom_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{edit_attention_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{viewer_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{image_viewer_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{zoom_image_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{tag_autocomplete_js_path}"></script>\n'
    head += f'<script type="text/javascript">window.tag_csv_files = {tag_files_json};</script>\n'
    head += f'<script type="text/javascript">window.chant_json_files = {chant_files_json};</script>\n'
    head += f'<script type="text/javascript">window.tac_user_config = {tac_cfg};</script>\n'
    head += f'<meta name="samples-path" content="{samples_path}">\n'

    if args_manager.args.theme:
        head += f'<script type="text/javascript">set_theme(\"{args_manager.args.theme}\");</script>\n'

    return head


def css_html():
    style_css_path = webpath('css/style.css')
    head = f'<link rel="stylesheet" property="stylesheet" href="{style_css_path}">'
    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
