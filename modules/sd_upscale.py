import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import threading
from concurrent.futures import ThreadPoolExecutor

from modules import default_pipeline as pipeline
from modules import core, config
import gc
import torch

from PIL import Image
import numpy as np
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


def apply_denoising(tile: Image.Image, prompt: str, denoising_strength: float, image_seed: int | None = None) -> Image.Image:
    """Apply Fooocus diffusion on a single tile using ``prompt`` and
    ``denoising_strength``. This mirrors the behaviour of features such as
    ``Vary`` and ``Upscale" in the official pipeline."""

    import numpy as np
    import random

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
    if image_seed is None:
        image_seed = random.randint(0, 2**32 - 1)

    images = pipeline.process_diffusion(
        positive_cond=positive_cond,
        negative_cond=negative_cond,
        steps=20,
        switch=0,
        width=w * 8,
        height=h * 8,
        image_seed=image_seed,
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
    blend_mask: Optional[np.ndarray] = None


def create_blend_mask(tile_w: int, tile_h: int, overlap: int) -> np.ndarray:
    """Create a feathered blend mask used when recombining tiles."""
    mask = np.ones((tile_h, tile_w), dtype=np.float32)
    feather_size = min(overlap // 2, 32)
    if feather_size > 0:
        for i in range(feather_size):
            mask[i, :] *= i / feather_size
            mask[-(i + 1), :] *= i / feather_size
            mask[:, i] *= i / feather_size
            mask[:, -(i + 1)] *= i / feather_size
    return mask


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    w, h = image.size
    grid = Grid(image_w=w, image_h=h, tile_w=tile_w, tile_h=tile_h, overlap=overlap, tiles=[])

    cols = max(math.ceil((w - overlap) / float(tile_w - overlap)), 1)
    rows = max(math.ceil((h - overlap) / float(tile_h - overlap)), 1)

    dx = (w - tile_w) / max(cols - 1, 1) if cols > 1 else 0
    dy = (h - tile_h) / max(rows - 1, 1) if rows > 1 else 0

    grid.blend_mask = create_blend_mask(tile_w, tile_h, overlap)

    print(f'[Future-Sd-Upscale] Splitting image into {rows}x{cols} tiles ' \
          f'({tile_w}x{tile_h}, overlap={overlap})')

    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        row_images = []
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)
            tile = image.crop((x, y, x + tile_w, y + tile_h))
            row_images.append([x, tile_w, tile])
        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid_seamless(grid: Grid, upscaled_tiles: List[List[Image.Image]], scale_factor: float) -> Image.Image:
    """Combine tiles using feathered blending to hide seams."""
    dst_w = int(grid.image_w * scale_factor)
    dst_h = int(grid.image_h * scale_factor)

    combined_array = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
    weight_array = np.zeros((dst_h, dst_w), dtype=np.float32)

    try:
        scaled_mask = np.array(
            Image.fromarray(grid.blend_mask).resize(
                (int(grid.tile_w * scale_factor), int(grid.tile_h * scale_factor)),
                Image.LANCZOS,
            )
        )
    except Exception:
        scaled_mask = np.ones(
            (int(grid.tile_h * scale_factor), int(grid.tile_w * scale_factor)),
            dtype=np.float32,
        )

    if scaled_mask.ndim == 0:
        scaled_mask = np.ones(
            (int(grid.tile_h * scale_factor), int(grid.tile_w * scale_factor)),
            dtype=np.float32,
        )

    for row_idx, (y, th, row) in enumerate(grid.tiles):
        for col_idx, (x, tw, _) in enumerate(row):
            dst_x = int(x * scale_factor)
            dst_y = int(y * scale_factor)
            dst_tw = int(tw * scale_factor)
            dst_th = int(th * scale_factor)

            tile_img = upscaled_tiles[row_idx][col_idx]
            if tile_img is None:
                tile_img = Image.new('RGB', (dst_tw, dst_th), (0, 0, 0))
            tile_array = np.array(tile_img).astype(np.float32)
            if tile_array.ndim == 2:
                tile_array = np.stack([tile_array] * 3, axis=2)

            if dst_y >= dst_h or dst_x >= dst_w:
                continue

            y_end = min(dst_y + dst_th, dst_h)
            x_end = min(dst_x + dst_tw, dst_w)

            mask_h = y_end - dst_y
            mask_w = x_end - dst_x
            tile_array = tile_array[:mask_h, :mask_w]
            current_mask = scaled_mask[:mask_h, :mask_w]
            if current_mask.ndim == 0:
                current_mask = np.ones((mask_h, mask_w), dtype=np.float32)
            elif current_mask.ndim == 3:
                current_mask = current_mask[:, :, 0]
            elif current_mask.ndim != 2:
                current_mask = np.reshape(current_mask, (mask_h, mask_w))

            combined_array[dst_y:y_end, dst_x:x_end] += tile_array[:mask_h, :mask_w] * current_mask[:, :, np.newaxis]
            weight_array[dst_y:y_end, dst_x:x_end] += current_mask

    weight_array = np.maximum(weight_array, 1e-8)
    combined_array /= weight_array[:, :, np.newaxis]

    return Image.fromarray(np.clip(combined_array, 0, 255).astype(np.uint8))


class TileProcessor:
    """Thread-safe tile processor with optional ESRGAN upscale."""

    def __init__(self, upscaler_name: str, max_memory_tiles: int = 8):
        self.upscaler_name = upscaler_name
        self.max_memory_tiles = max_memory_tiles
        self._lock = threading.Lock()

    def process_batch(self, batch_images: List[np.ndarray], batch_info: List[Tuple]) -> List[np.ndarray]:
        if not batch_images:
            return []

        try:
            if self.upscaler_name != "None":
                from modules.upscaler import perform_upscale
                with self._lock:
                    upscaled = perform_upscale(np.stack(batch_images), self.upscaler_name)
            else:
                from modules.util import resample_image
                upscaled = [resample_image(img, info[4], info[5]) for img, info in zip(batch_images, batch_info)]
        except Exception as e:
            print(f"[Future-Sd-Upscale] ESRGAN failed: {e}. Falling back to Lanczos resize.")
            from modules.util import resample_image
            upscaled = [resample_image(img, info[4], info[5]) for img, info in zip(batch_images, batch_info)]

        if isinstance(upscaled, np.ndarray):
            return list(upscaled)
        return upscaled


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
        progress_callback: Optional[Callable] = None,
        prompt: str = "",
        denoising_strength: float = 0.0,
        seed: Optional[int] = None,
        num_threads: int = 2,
) -> Image.Image:
    """Optimized upscale with seamless blending and parallel processing."""

    print(
        f'[Future-Sd-Upscale] Starting optimized upscale: factor={scale_factor}, '
        f'tile_size={tile_size}, overlap={overlap}, model={upscaler_name}, '
        f'batch={batch_size}, threads={num_threads}'
    )

    effective_overlap = max(overlap, 32)
    if effective_overlap != overlap:
        print(f'[Future-Sd-Upscale] Increased overlap from {overlap} to {effective_overlap} for better blending')

    grid = split_grid(image, tile_w=tile_size, tile_h=tile_size, overlap=effective_overlap)
    total_tiles = sum(len(r[2]) for r in grid.tiles)
    done_tiles = 0

    upscaled_tiles = [[None for _ in row[2]] for row in grid.tiles]

    processor = TileProcessor(upscaler_name, max_memory_tiles=batch_size * 2)

    all_tiles = []
    for row_idx, (y, th, row) in enumerate(grid.tiles):
        for col_idx, (x, tw, tile) in enumerate(row):
            dst_tw = int(tw * scale_factor)
            dst_th = int(th * scale_factor)
            all_tiles.append((row_idx, col_idx, np.array(tile), dst_tw, dst_th))

    def process_tile_batch(batch_start: int, batch_end: int):
        nonlocal done_tiles
        batch_tiles = all_tiles[batch_start:batch_end]

        if not batch_tiles:
            return

        batch_images = [tile_data[2] for tile_data in batch_tiles]
        batch_info = [(td[0], td[1], 0, 0, td[3], td[4]) for td in batch_tiles]

        try:
            results = processor.process_batch(batch_images, batch_info)
        except Exception as e:
            print(f"[Future-Sd-Upscale] Batch failed: {e}")
            results = [np.zeros((info[4], info[5], 3), dtype=np.uint8) for info in batch_info]

        for (row_idx, col_idx, _, dst_tw, dst_th), result in zip(batch_tiles, results):
            try:
                tile_img = Image.fromarray(result)
            except Exception:
                tile_img = Image.new('RGB', (dst_tw, dst_th), (0, 0, 0))

            if denoising_strength > 0:
                tile_img = apply_denoising(tile_img, prompt, denoising_strength, image_seed=seed)

            from modules.util import resample_image
            tile_np = resample_image(np.array(tile_img), width=dst_tw, height=dst_th)
            tile_img = Image.fromarray(tile_np)

            upscaled_tiles[row_idx][col_idx] = tile_img
            done_tiles += 1

            if progress_callback is not None and (
                done_tiles == total_tiles or done_tiles % max(1, total_tiles // 10) == 0
            ):
                temp_combined = combine_grid_seamless(grid, upscaled_tiles, scale_factor)
                progress_callback(done_tiles, total_tiles, temp_combined)

    batch_ranges = [(i, min(i + batch_size, len(all_tiles))) for i in range(0, len(all_tiles), batch_size)]

    if num_threads > 1 and len(batch_ranges) > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_tile_batch, start, end) for start, end in batch_ranges]
            for i, future in enumerate(futures):
                future.result()
                if i % 2 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    else:
        for start, end in batch_ranges:
            process_tile_batch(start, end)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    combined_image = combine_grid_seamless(grid, upscaled_tiles, scale_factor)

    print(
        f'[Future-Sd-Upscale] Finished optimized upscale. Result size: '
        f'{combined_image.size}'
    )
    return combined_image
