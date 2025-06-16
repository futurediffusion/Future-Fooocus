# Repository Structure

- `modules/`: Core application modules and utilities.
  - `NewLoraSystem/`: Implementation of the new LoRA system.
  - `lora_utils.py`: Helper functions for locating LoRA files and handling metadata.
- `models/`: Default model locations such as `loras`, `checkpoints`, etc.
- `ldm_patched/`: Patched Stable Diffusion components.
- `extras/`: Additional features including BLIP, expansion, and preprocessing tools.
- `javascript/` and `css/`: Front‑end scripts and styles.
- `tests/`: Unit tests for utility modules.
- Other project files (`webui.py`, `launch.py`, etc.) start the application or configure runtime.
