gradio_root = None
prompt_styles = None

import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def html(name: str) -> str:
    """Return contents of an HTML template bundled with the repo."""
    paths = [
        os.path.join(_MODULE_DIR, "NewLoraSystem", "html", name),
        os.path.join(_MODULE_DIR, "html", name),
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(name)

