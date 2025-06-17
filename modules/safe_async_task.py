import os
from modules.async_worker import AsyncTask
from modules.lora_utils import get_lora_path


class SafeAsyncTask(AsyncTask):
    """AsyncTask wrapper that validates LoRA paths before processing."""

    def __init__(self, args):
        super().__init__(args)

        validated = []
        for name, weight in self.loras:
            if name in {"", "None"}:
                validated.append((name, weight))
                continue

            path = get_lora_path(name)
            if path and os.path.isfile(path):
                validated.append((path, weight))
            else:
                print(f"[SafeAsyncTask] Warning: LoRA file not found: {name}")
        self.loras = validated
