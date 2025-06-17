import os
from typing import Any, Dict

try:
    from safetensors.torch import safe_open
except Exception:
    safe_open = None


def read_metadata_from_safetensors(filename: str) -> Dict[str, Any]:
    """Read metadata from a safetensors file.

    Returns an empty dict if reading fails or safetensors is unavailable."""
    if safe_open is None or not os.path.isfile(filename):
        return {}
    try:
        with safe_open(filename, framework="pt", device="cpu") as f:
            return dict(f.metadata())
    except Exception:
        return {}


class _ModelData:
    """Minimal stub to satisfy modules depending on sd_models."""

    def get_sd_model(self):
        """Return the current stable diffusion model if available."""
        return None


model_data = _ModelData()
