import math
from dataclasses import dataclass
from typing import List

from modules import default_pipeline as pipeline
from modules import core, config
import gc
import torch

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
        batch_size: int = 4,
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
        Name of the ESRGAN model to use. ``"None"`` disables the model and only
        performs a Lanczos resize.
    batch_size : int, optional
        Number of tiles processed simultaneously, by default ``4``.
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
        f'tile_size={tile_size}, overlap={overlap}, model={upscaler_name}, '
        f'batch={batch_size}'
    )

    # do not resize the whole image beforehand

    grid = split_grid(image, tile_w=tile_size, tile_h=tile_size, overlap=overlap)
    total_tiles = sum(len(r[2]) for r in grid.tiles)
    done_tiles = 0
    dst_w = int(grid.image_w * scale_factor)
    dst_h = int(grid.image_h * scale_factor)
    combined_image = Image.new('RGB', (dst_w, dst_h))

    batch_images = []
    batch_info = []
    batch_counter = 0

    def process_batch():
        nonlocal batch_images, batch_info, batch_counter, done_tiles
        if not batch_images:
            return
        try:
            if upscaler_name != "None":
                upscaled = perform_upscale(np.stack(batch_images), upscaler_name)
            else:
                upscaled = [resample_image(img, info[4], info[5]) for img, info in zip(batch_images, batch_info)]
        except Exception as e:
            print(f"[Future-Sd-Upscale] ESRGAN failed: {e}. Falling back to Lanczos resize.")
            upscaled = [resample_image(img, info[4], info[5]) for img, info in zip(batch_images, batch_info)]

        if isinstance(upscaled, np.ndarray):
            results = list(upscaled)
        else:
            results = upscaled

        for (row_idx, col_idx, dx, dy, d_tw, d_th), out_np in zip(batch_info, results):
            tile_img = Image.fromarray(out_np)
            if denoising_strength > 0:
                tile_img = apply_denoising(tile_img, prompt, denoising_strength)
            tile_np = np.array(tile_img)
            tile_np = resample_image(tile_np, width=d_tw, height=d_th)
            tile_img = Image.fromarray(tile_np)
            combined_image.paste(tile_img.crop((0, 0, d_tw, d_th)), (dx, dy))
            done_tiles += 1
            if progress_callback is not None:
                progress_callback(done_tiles, total_tiles, combined_image)
            grid.tiles[row_idx][2][col_idx][2] = tile_img

        batch_images = []
        batch_info = []
        batch_counter += 1
        if batch_counter % 3 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for row_index, (y, th, row) in enumerate(grid.tiles):
        for col_index, (x, tw, tile) in enumerate(row):
            dst_x = int(x * scale_factor)
            dst_y = int(y * scale_factor)
            dst_tw = int(tw * scale_factor)
            dst_th = int(th * scale_factor)

            batch_images.append(np.array(tile))
            batch_info.append((row_index, col_index, dst_x, dst_y, dst_tw, dst_th))

            if len(batch_images) >= batch_size:
                process_batch()

    process_batch()

    print(
        f'[Future-Sd-Upscale] Finished upscale. Result size: '
        f'{combined_image.size}'
    )
    return combined_image
