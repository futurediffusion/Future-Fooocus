import math
from dataclasses import dataclass
from typing import List

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
    """

    import numpy as np
    from modules.upscaler import perform_upscale
    from modules.util import resample_image, LANCZOS

    # resize the whole image first so tiling operates on the final resolution
    if scale_factor != 1.0:
        w = int(image.width * scale_factor)
        h = int(image.height * scale_factor)
        image = image.resize((w, h), resample=LANCZOS)

    grid = split_grid(image, tile_w=tile_size, tile_h=tile_size, overlap=overlap)

    for row_index, (_, th, row) in enumerate(grid.tiles):
        for col_index, (x, tw, tile) in enumerate(row):
            tile_np = np.array(tile)
            if upscaler_name != "None":
                tile_np = perform_upscale(tile_np)
            # return tile to its original size so the grid can be recombined
            tile_np = resample_image(tile_np, width=tw, height=th)
            row[col_index][2] = Image.fromarray(tile_np)

    combined = combine_grid(grid)
    return combined
