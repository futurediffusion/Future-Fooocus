import gradio as gr


def create_refresh_button(components, refresh_func, update_func, elem_id=None):
    """Create a small refresh button that updates given components."""
    btn = gr.Button(value="\U0001f504", elem_id=elem_id, variant="secondary")

    def _refresh():
        refresh_func()
        update = update_func()
        if not isinstance(update, list):
            update = [update] * len(components)
        return [gr.update(**update_elem) if isinstance(update_elem, dict) else gr.update(**update) for update_elem in update]

    btn.click(_refresh, outputs=components, show_progress=False)
    return btn


def setup_dialog(button_show, dialog, button_close):
    """Simple show/hide behaviour for a dialog container."""
    button_show.click(lambda: gr.update(visible=True), outputs=dialog, show_progress=False)
    button_close.click(lambda: gr.update(visible=False), outputs=dialog, show_progress=False)
    return dialog
