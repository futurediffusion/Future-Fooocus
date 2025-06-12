import gradio as gr
from modules.util import join_prompts
import shared
from modules import config


def generar_prompt(personajes, nombres, lora_enable, lora_model, lora_weight,
                   angulo, expresion, pose, fondo, estilos, estilos_negativos):
    nombres_part = ', '.join([n.strip() for n in nombres.split(',') if n.strip()]) if nombres else ''

    lora_part = ''
    if lora_enable and lora_model != 'None':
        lora_part = f"<lora:{lora_model}:{lora_weight}>"

    base_prompt = join_prompts(
        personajes,
        nombres_part,
        lora_part,
        angulo,
        expresion,
        pose,
        fondo,
    )

    base_prompt = shared.prompt_styles.apply_styles_to_prompt(base_prompt, estilos)
    negativo = shared.prompt_styles.apply_negative_styles_to_prompt('', estilos_negativos)

    return gr.Textbox.update(value=base_prompt), gr.Textbox.update(value=negativo)


def create_ui(tabname: str, main_prompt: gr.Textbox, main_negative: gr.Textbox):
    with gr.Box(elem_id=f"{tabname}_prompt_builder"):
        gr.Markdown("### Prompt Builder")
        with gr.Row():
            personajes = gr.Dropdown(
                label="Personajes",
                choices=['1girl', '1boy', '2girls', '2boys'],
                value='1girl'
            )
            angulo = gr.Dropdown(
                label="Ángulo",
                choices=['frontal', 'lateral', '3/4 view', 'desde arriba', 'desde abajo', 'pov'],
                value='frontal'
            )
        with gr.Row():
            nombres = gr.Textbox(label='Nombres')
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
            expresiones = gr.Dropdown(
                label='Expresiones',
                choices=['feliz', 'triste', 'enojado', 'sorprendido', 'pensativo'],
                value='feliz'
            )
            poses = gr.Dropdown(
                label='Poses',
                choices=['de pie', 'sentado', 'corriendo', 'saltando', 'acostado'],
                value='de pie'
            )
        with gr.Row():
            fondo = gr.Dropdown(
                label='Fondo',
                choices=['bosque', 'playa', 'ciudad', 'interior', 'espacio'],
                value='bosque'
            )
        with gr.Row():
            estilos = gr.Dropdown(label='Estilos', choices=list(shared.prompt_styles.styles), multiselect=True)
            estilos_negativos = gr.Dropdown(label='Estilos negativos', choices=list(shared.prompt_styles.styles), multiselect=True)
        with gr.Row():
            generar = gr.Button('Generar')

        generar.click(
            fn=generar_prompt,
            inputs=[personajes, nombres, lora_enable, lora_model, lora_weight,
                    angulo, expresiones, poses, fondo, estilos, estilos_negativos],
            outputs=[main_prompt, main_negative],
            show_progress=False,
        )

    return {
        'personajes': personajes,
        'nombres': nombres,
        'lora_enable': lora_enable,
        'lora_model': lora_model,
        'lora_weight': lora_weight,
        'angulo': angulo,
        'expresiones': expresiones,
        'poses': poses,
        'fondo': fondo,
        'estilos': estilos,
        'estilos_negativos': estilos_negativos,
        'generar': generar,
    }
