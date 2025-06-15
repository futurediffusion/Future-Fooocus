import math
from dataclasses import dataclass
from typing import List

from modules.processing import StableDiffusionProcessingImg2Img, process_images

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


def apply_denoising(tile: Image.Image, prompt: str, denoising_strength: float):
    """Apply img2img diffusion on a single tile using the given prompt and
    denoising strength."""
    p = StableDiffusionProcessingImg2Img(
        init_images=[tile],
        prompt=prompt,
        seed=-1,
        steps=20,
        cfg_scale=7.5,
        width=tile.width,
        height=tile.height,
        denoising_strength=denoising_strength,
    )
    result = process_images(p)
    return result.images[0]


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
        Prompt used for denoising each tile when ``denoising_strength`` > 0.
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
            tile_np = np.array(tile)
            if upscaler_name != "None":
                tile_np = perform_upscale(tile_np)
            tile_np = resample_image(tile_np, width=tw, height=th)
            processed_tile = Image.fromarray(tile_np)
            if denoising_strength > 0:
                processed_tile = apply_denoising(processed_tile, prompt, denoising_strength)
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
