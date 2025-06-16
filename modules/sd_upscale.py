import math
from dataclasses import dataclass
from typing import List

from modules import default_pipeline as pipeline
from modules import core, config

from PIL import Image
from ldm_patched.utils import path_utils


def _find_upscalers():
    try:
        models = path_utils.get_filename_list("upscale_models")
    except Exception as e:
        print(f"Failed to load upscale models: {e}")
        models = []
    return ['None'] + models


DEFAULT_UPSCALERS = _find_upscalers()


def reload_upscalers() -> List[str]:
    """Reload available upscale models from disk."""
    global DEFAULT_UPSCALERS
    DEFAULT_UPSCALERS = _find_upscalers()
    return DEFAULT_UPSCALERS


def apply_denoising(tile: Image.Image, prompt: str, denoising_strength: float) -> Image.Image:
    """Apply Fooocus diffusion on a single tile using ``prompt`` and
    ``denoising_strength``. This mirrors the behaviour of features such as
    ``Vary`` and ``Upscale" in the official pipeline."""

    import numpy as np

    # Encode prompt and default negative prompt using Fooocus CLIP pipeline
    positive_cond = pipeline.clip_encode(texts=[prompt], pool_top_k=1)
    negative_prompt = config.default_prompt_negative or ""
    negative_cond = pipeline.clip_encode(texts=[negative_prompt], pool_top_k=1)

    # Prepare latent from image using the currently loaded VAE
    candidate_vae, _ = pipeline.get_candidate_vae(
        steps=20, switch=0, denoise=denoising_strength, refiner_swap_method="joint"
    )
    tile_tensor = core.numpy_to_pytorch(np.array(tile))
    latent = core.encode_vae(vae=candidate_vae, pixels=tile_tensor, tiled=False)
    _, _, h, w = latent["samples"].shape

    # Run diffusion on the tile latent
    images = pipeline.process_diffusion(
        positive_cond=positive_cond,
        negative_cond=negative_cond,
        steps=20,
        switch=0,
        width=w * 8,
        height=h * 8,
        image_seed=0,
        callback=None,
        sampler_name=config.default_sampler,
        scheduler_name=config.default_scheduler,
        latent=latent,
        denoise=denoising_strength,
        tiled=False,
        cfg_scale=config.default_cfg_scale,
        refiner_swap_method="joint",
        disable_preview=True,
    )

    return Image.fromarray(images[0])


@dataclass
class Grid:
    image_w: int
    image_h: int
    tile_w: int
    tile_h: int
    overlap: int
    tiles: List


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    w, h = image.size
    grid = Grid(image_w=w, image_h=h, tile_w=tile_w, tile_h=tile_h, overlap=overlap, tiles=[])
    cols = max(math.ceil((w - overlap) / float(tile_w - overlap)), 1)
    rows = max(math.ceil((h - overlap) / float(tile_h - overlap)), 1)
    dx = (w - tile_w) / max(cols - 1, 1)
    dy = (h - tile_h) / max(rows - 1, 1)
    print(f'[Future-Sd-Upscale] Splitting image into {rows}x{cols} tiles ' \
          f'({tile_w}x{tile_h}, overlap={overlap})')
    for row in range(rows):
        y = int(row * dy)
        row_images = []
        for col in range(cols):
            x = int(col * dx)
            tile = image.crop((x, y, x + tile_w, y + tile_h))
            row_images.append([x, tile_w, tile])
        grid.tiles.append([y, tile_h, row_images])
    return grid


def combine_grid(grid: Grid) -> Image.Image:
    combined_image = Image.new('RGB', (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        for x, w, tile in row:
            combined_image.paste(tile.crop((0, 0, w, h)), (x, y))
    return combined_image


def upscale_image(
        image: Image.Image,
        overlap: int,
        scale_factor: float,
        tile_size: int = 512,
        upscaler_name: str = "None",
        progress_callback=None,
        prompt: str = "",
        denoising_strength: float = 0.0,
) -> Image.Image:
    """Upscale ``image`` by ``scale_factor`` while processing tiles individually.

    This helper is used by ``async_worker`` when the *SD Upscale* checkbox is
    enabled.  The original implementation was a placeholder that merely resized
    the input image.  This version performs a real tile based upscale so the
    input image and overlap values have visible effect.

    Parameters
    ----------
    image : PIL.Image
        Image to be upscaled.
    overlap : int
        Overlap size between tiles.
    scale_factor : float
        Overall scaling factor for the image.
    tile_size : int, optional
        Size of each tile processed individually, by default ``512``.
    upscaler_name : str, optional
        Name of the ESRGAN model to use.  ``"None"`` disables the model and only
        performs a Lanczos resize.
    progress_callback : callable, optional
        Called after each tile is processed with ``(done_tiles, total_tiles,
        preview_image)``.
    prompt : str, optional
        Prompt used for denoising each tile when ``denoising_strength > 0``.
    denoising_strength : float, optional
        Strength of the img2img denoising applied to each tile.
    """

    import numpy as np
    from modules.upscaler import perform_upscale
    from modules.util import resample_image, LANCZOS

    print(
        f'[Future-Sd-Upscale] Starting upscale: factor={scale_factor}, '
        f'tile_size={tile_size}, overlap={overlap}, model={upscaler_name}'
    )

    # resize the whole image first so tiling operates on the final resolution
    if scale_factor != 1.0:
        w = int(image.width * scale_factor)
        h = int(image.height * scale_factor)
        image = image.resize((w, h), resample=LANCZOS)

    grid = split_grid(image, tile_w=tile_size, tile_h=tile_size, overlap=overlap)
    total_tiles = sum(len(r[2]) for r in grid.tiles)
    done_tiles = 0
    combined_image = Image.new('RGB', (grid.image_w, grid.image_h))

    for row_index, (y, th, row) in enumerate(grid.tiles):
        for col_index, (x, tw, tile) in enumerate(row):
            processed_tile = tile

            # 1. Optional ESRGAN upscale on the raw tile
            if upscaler_name != "None":
                tile_np = perform_upscale(np.array(processed_tile))
                processed_tile = Image.fromarray(tile_np)

            # 2. Apply denoising using the Fooocus pipeline
            if denoising_strength > 0:
                processed_tile = apply_denoising(processed_tile, prompt, denoising_strength)

            # 3. Resize back to the expected tile size
            tile_np = np.array(processed_tile)
            tile_np = resample_image(tile_np, width=tw, height=th)
            processed_tile = Image.fromarray(tile_np)

            # 4. Paste into the final combined image
            combined_image.paste(processed_tile.crop((0, 0, tw, th)), (x, y))
            done_tiles += 1
            if progress_callback is not None:
                progress_callback(done_tiles, total_tiles, combined_image)
            row[col_index][2] = processed_tile

    print(
        f'[Future-Sd-Upscale] Finished upscale. Result size: '
        f'{combined_image.size}'
    )
    return combined_image
