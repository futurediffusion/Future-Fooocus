import math
from dataclasses import dataclass
from typing import List

from PIL import Image

DEFAULT_UPSCALERS = ['None']


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


def upscale_image(image: Image.Image, overlap: int, scale_factor: float, tile_size: int = 512) -> Image.Image:
    if scale_factor != 1.0:
        w = int(image.width * scale_factor)
        h = int(image.height * scale_factor)
        image = image.resize((w, h), resample=Image.LANCZOS)
    grid = split_grid(image, tile_w=tile_size, tile_h=tile_size, overlap=overlap)
    # Placeholder processing: normally each tile would be sent through diffusion
    combined = combine_grid(grid)
    return combined
