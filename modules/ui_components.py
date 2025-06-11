import gradio as gr


def ToolButton(value, elem_id=None, tooltip=None, **kwargs):
    """Simple button used as a small tool icon."""
    return gr.Button(
        value=value,
        elem_id=elem_id,
        tooltip=tooltip,
        variant="secondary",
        **kwargs,
    )
