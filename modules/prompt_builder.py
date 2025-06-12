import gradio as gr
from modules.util import join_prompts
import shared


def generar_prompt(num_personajes, nombres, angulo, expresiones, poses, texto_extra, estilos, estilos_negativos):
    nombres_part = ', '.join([n.strip() for n in nombres.split(',') if n.strip()]) if nombres else ''
    expresiones_part = ', '.join(expresiones) if expresiones else ''
    poses_part = ', '.join(poses) if poses else ''

    base_prompt = join_prompts(
        f"{num_personajes} personaje" + ('' if num_personajes == 1 else 's'),
        nombres_part,
        angulo,
        expresiones_part,
        poses_part,
        texto_extra,
    )

    base_prompt = shared.prompt_styles.apply_styles_to_prompt(base_prompt, estilos)
    negativo = shared.prompt_styles.apply_negative_styles_to_prompt('', estilos_negativos)

    return gr.Textbox.update(value=base_prompt), gr.Textbox.update(value=negativo)


def create_ui(tabname: str, main_prompt: gr.Textbox, main_negative: gr.Textbox):
    with gr.Box(elem_id=f"{tabname}_prompt_builder"):
        gr.Markdown("### Prompt Builder")
        with gr.Row():
            num_personajes = gr.Dropdown(label="Cantidad de personajes", choices=[1, 2, 3, 4], value=1)
            angulo = gr.Dropdown(label="Ángulo", choices=['frontal', 'lateral', 'espalda'], value='frontal')
            cfg = gr.Slider(label='CFG', minimum=1.0, maximum=20.0, step=0.5, value=7.0)
        with gr.Row():
            nombres = gr.Textbox(label='Nombres')
        with gr.Row():
            expresiones = gr.CheckboxGroup(label='Expresiones', choices=['feliz', 'triste', 'enojado'])
            poses = gr.CheckboxGroup(label='Poses', choices=['de pie', 'sentado', 'corriendo'])
        with gr.Row():
            texto_extra = gr.Textbox(label='Texto adicional')
        with gr.Row():
            estilos = gr.Dropdown(label='Estilos', choices=list(shared.prompt_styles.styles), multiselect=True)
            estilos_negativos = gr.Dropdown(label='Estilos negativos', choices=list(shared.prompt_styles.styles), multiselect=True)
        with gr.Row():
            generar = gr.Button('Generar')
            copiar = gr.Button('Copiar')

        generar.click(
            fn=generar_prompt,
            inputs=[num_personajes, nombres, angulo, expresiones, poses, texto_extra, estilos, estilos_negativos],
            outputs=[main_prompt, main_negative],
            show_progress=False,
        )

        copiar.click(
            fn=None,
            _js=f"() => navigator.clipboard.writeText(document.getElementById('{main_prompt.elem_id}').value)",
            inputs=None,
            outputs=None,
        )

    return {
        'num_personajes': num_personajes,
        'nombres': nombres,
        'angulo': angulo,
        'expresiones': expresiones,
        'poses': poses,
        'texto_extra': texto_extra,
        'estilos': estilos,
        'estilos_negativos': estilos_negativos,
        'cfg': cfg,
        'generar': generar,
        'copiar': copiar,
    }
