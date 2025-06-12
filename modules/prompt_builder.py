import gradio as gr
from modules.util import join_prompts
import shared
from modules import config


def generar_prompt(characters, names, lora_enable, lora_model, lora_weight,
                   angle, expression, pose, background):
    names_part = ', '.join([n.strip() for n in names.split(',') if n.strip()]) if names else ''

    lora_part = ''
    if lora_enable and lora_model != 'None':
        lora_part = f"<lora:{lora_model}:{lora_weight}>"

    base_prompt = join_prompts(
        characters,
        names_part,
        lora_part,
        angle,
        expression,
        pose,
        background,
    )

    return gr.Textbox.update(value=base_prompt), gr.Textbox.update(value='')


def create_ui(tabname: str, main_prompt: gr.Textbox, main_negative: gr.Textbox):
    with gr.Accordion(label='Prompt Builder', open=False, elem_id=f"{tabname}_prompt_builder"):
        with gr.Row():
            characters = gr.Dropdown(
                label="Characters",
                choices=['1girl', '1boy', '2girls', '2boys'],
                value='1girl'
            )
            angle = gr.Dropdown(
                label="Angle",
                choices=['front view', 'side view', '3/4 view', 'from above', 'from below', 'POV'],
                value='front view'
            )
        with gr.Row():
            names = gr.Textbox(label='Names')
        with gr.Row():
            lora_enable = gr.Checkbox(label='LoRA Enable', value=False)
            lora_model = gr.Dropdown(
                label='LoRA',
                choices=['None'] + config.lora_filenames,
                value='None'
            )
            lora_weight = gr.Slider(
                label='Weight',
                minimum=config.default_loras_min_weight,
                maximum=config.default_loras_max_weight,
                step=0.01,
                value=1.0
            )
        with gr.Row():
            expressions = gr.Dropdown(
                label='Expressions',
                choices=['happy', 'sad', 'angry', 'surprised', 'thoughtful'],
                value='happy'
            )
            poses = gr.Dropdown(
                label='Poses',
                choices=['standing', 'sitting', 'running', 'jumping', 'lying down'],
                value='standing'
            )
        with gr.Row():
            background = gr.Dropdown(
                label='Background',
                choices=['forest', 'beach', 'city', 'indoor', 'space'],
                value='forest'
            )
        with gr.Row():
            generate = gr.Button('Generate')

        generate.click(
            fn=generar_prompt,
            inputs=[characters, names, lora_enable, lora_model, lora_weight,
                    angle, expressions, poses, background],
            outputs=[main_prompt, main_negative],
            show_progress=False,
        )

    return {
        'characters': characters,
        'names': names,
        'lora_enable': lora_enable,
        'lora_model': lora_model,
        'lora_weight': lora_weight,
        'angle': angle,
        'expressions': expressions,
        'poses': poses,
        'background': background,
        'generate': generate,
    }
