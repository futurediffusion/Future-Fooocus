import math
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
import numpy as np
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
    global DEFAULT_UPSCALERS
    DEFAULT_UPSCALERS = _find_upscalers()
    return DEFAULT_UPSCALERS


@dataclass
class TileInfo:
    """Optimized tile information structure."""
    src_x: int
    src_y: int
    src_w: int
    src_h: int
    dst_x: int
    dst_y: int
    dst_w: int
    dst_h: int
    overlap_mask: Optional[np.ndarray] = None


class OptimizedUpscaler:
    """Fooocus optimized upscaler with smart tiling and caching."""

    def __init__(self):
        self._cached_model = None
        self._cached_model_name = None
        self._device = self._get_optimal_device()

    def _get_optimal_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _load_model_cached(self, upscaler_name: str):
        if upscaler_name == "None":
            return None
        if self._cached_model_name != upscaler_name:
            if self._cached_model is not None:
                del self._cached_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            try:
                from modules.upscaler import load_upscaler_model
                self._cached_model = load_upscaler_model(upscaler_name, self._device)
                self._cached_model_name = upscaler_name
                print(f"[FooocusUpscale] Loaded model: {upscaler_name}")
            except Exception as e:
                print(f"[FooocusUpscale] Failed to load {upscaler_name}: {e}")
                self._cached_model = None
                self._cached_model_name = None
        return self._cached_model

    def _calculate_smart_tiles(self, width: int, height: int, tile_size: int,
                              overlap: int, scale_factor: float) -> List[TileInfo]:
        tiles = []
        cols = max(math.ceil((width - overlap) / (tile_size - overlap)), 1)
        rows = max(math.ceil((height - overlap) / (tile_size - overlap)), 1)
        if cols > 1:
            dx = (width - tile_size) / (cols - 1)
        else:
            dx = 0
        if rows > 1:
            dy = (height - tile_size) / (rows - 1)
        else:
            dy = 0
        print(f"[FooocusUpscale] Grid: {rows}x{cols}, tile_size: {tile_size}, overlap: {overlap}")
        for row in range(rows):
            for col in range(cols):
                src_x = int(col * dx)
                src_y = int(row * dy)
                src_w = min(tile_size, width - src_x)
                src_h = min(tile_size, height - src_y)
                dst_x = int(src_x * scale_factor)
                dst_y = int(src_y * scale_factor)
                dst_w = int(src_w * scale_factor)
                dst_h = int(src_h * scale_factor)
                overlap_mask = self._create_overlap_mask(
                    src_w, src_h, overlap, row, col, rows, cols
                ) if overlap > 0 else None
                tiles.append(TileInfo(
                    src_x=src_x, src_y=src_y, src_w=src_w, src_h=src_h,
                    dst_x=dst_x, dst_y=dst_y, dst_w=dst_w, dst_h=dst_h,
                    overlap_mask=overlap_mask
                ))
        return tiles

    def _create_overlap_mask(self, w: int, h: int, overlap: int,
                            row: int, col: int, rows: int, cols: int) -> np.ndarray:
        mask = np.ones((h, w), dtype=np.float32)
        fade_size = min(overlap, w // 4, h // 4)
        if fade_size > 0:
            if col > 0:
                for i in range(fade_size):
                    mask[:, i] *= i / fade_size
            if col < cols - 1:
                for i in range(fade_size):
                    mask[:, w - 1 - i] *= i / fade_size
            if row > 0:
                for i in range(fade_size):
                    mask[i, :] *= i / fade_size
            if row < rows - 1:
                for i in range(fade_size):
                    mask[h - 1 - i, :] *= i / fade_size
        return mask

    def _process_tile_batch(self, tiles_data: List[Tuple[np.ndarray, TileInfo]],
                           upscaler_name: str) -> List[Tuple[np.ndarray, TileInfo]]:
        model = self._load_model_cached(upscaler_name)
        if model is None:
            from modules.util import resample_image
            results = []
            for tile_np, info in tiles_data:
                resized = resample_image(tile_np, info.dst_w, info.dst_h)
                results.append((resized, info))
            return results
        try:
            from modules.upscaler import upscale_with_model
            results = []
            for tile_np, info in tiles_data:
                if tile_np.dtype != np.uint8:
                    tile_np = (tile_np * 255).astype(np.uint8)
                upscaled_np = upscale_with_model(model, tile_np, self._device)
                if upscaled_np.shape[:2] != (info.dst_h, info.dst_w):
                    from modules.util import resample_image
                    upscaled_np = resample_image(upscaled_np, info.dst_w, info.dst_h)
                results.append((upscaled_np, info))
            return results
        except Exception as e:
            print(f"[FooocusUpscale] Model processing failed: {e}")
            from modules.util import resample_image
            results = []
            for tile_np, info in tiles_data:
                resized = resample_image(tile_np, info.dst_w, info.dst_h)
                results.append((resized, info))
            return results

    def _blend_tile_with_overlap(self, canvas: np.ndarray, tile: np.ndarray,
                                tile_info: TileInfo, canvas_weight: np.ndarray):
        dst_x, dst_y = tile_info.dst_x, tile_info.dst_y
        dst_h, dst_w = tile.shape[:2]
        end_y = min(dst_y + dst_h, canvas.shape[0])
        end_x = min(dst_x + dst_w, canvas.shape[1])
        actual_h = end_y - dst_y
        actual_w = end_x - dst_x
        if actual_h <= 0 or actual_w <= 0:
            return
        canvas_region = canvas[dst_y:end_y, dst_x:end_x]
        tile_region = tile[:actual_h, :actual_w]
        weight_region = canvas_weight[dst_y:end_y, dst_x:end_x]
        if tile_info.overlap_mask is not None:
            mask = tile_info.overlap_mask[:actual_h, :actual_w]
            if len(canvas_region.shape) == 3:
                mask = mask[:, :, np.newaxis]
            total_weight = weight_region + mask
            np.divide(weight_region * canvas_region + mask * tile_region,
                     total_weight, out=canvas_region, where=total_weight > 0)
            weight_region += mask.squeeze() if len(mask.shape) == 3 else mask
        else:
            canvas_region[:] = tile_region
            if len(weight_region.shape) == 2:
                weight_region[:] = 1.0
            else:
                weight_region[:] = 1.0

    def upscale_image(self, image: Image.Image, overlap: int, scale_factor: float,
                     tile_size: int = 512, upscaler_name: str = "None",
                     progress_callback: Optional[Callable] = None,
                     batch_size: int = 4) -> Image.Image:
        w, h = image.size
        final_w = int(w * scale_factor)
        final_h = int(h * scale_factor)
        print(f"[FooocusUpscale] {w}x{h} -> {final_w}x{final_h} (scale: {scale_factor}, model: {upscaler_name})")
        tiles = self._calculate_smart_tiles(w, h, tile_size, overlap, scale_factor)
        total_tiles = len(tiles)
        canvas = np.zeros((final_h, final_w, 3), dtype=np.float32)
        canvas_weight = np.zeros((final_h, final_w), dtype=np.float32)
        image_np = np.array(image)
        processed_count = 0
        for batch_start in range(0, total_tiles, batch_size):
            batch_end = min(batch_start + batch_size, total_tiles)
            batch_tiles = tiles[batch_start:batch_end]
            batch_data = []
            for info in batch_tiles:
                tile_np = image_np[
                    info.src_y:info.src_y + info.src_h,
                    info.src_x:info.src_x + info.src_w
                ].copy()
                batch_data.append((tile_np, info))
            try:
                processed_batch = self._process_tile_batch(batch_data, upscaler_name)
                for processed_tile, info in processed_batch:
                    self._blend_tile_with_overlap(canvas, processed_tile, info, canvas_weight)
                    processed_count += 1
                    if progress_callback:
                        preview = None
                        if processed_count % 5 == 0:
                            preview_canvas = np.clip(canvas, 0, 255).astype(np.uint8)
                            preview = Image.fromarray(preview_canvas)
                        progress_callback(processed_count, total_tiles, preview)
            except Exception as e:
                print(f"[FooocusUpscale] Batch processing error: {e}")
                processed_count += len(batch_tiles)
                if progress_callback:
                    progress_callback(processed_count, total_tiles, None)
            if batch_start % (batch_size * 3) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        final_canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(final_canvas)
        del canvas, canvas_weight, image_np
        gc.collect()
        print(f"[FooocusUpscale] Completed! Result: {result_image.size}")
        return result_image


_upscaler_instance = OptimizedUpscaler()


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
    return _upscaler_instance.upscale_image(
        image,
        overlap,
        scale_factor,
        tile_size,
        upscaler_name,
        progress_callback,
    )


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
    return Grid(image_w=w, image_h=h, tile_w=tile_w, tile_h=tile_h, overlap=overlap, tiles=[])


def combine_grid(grid: Grid) -> Image.Image:
    return Image.new('RGB', (grid.image_w, grid.image_h))
