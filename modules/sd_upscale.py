import math
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
import numpy as np
from PIL import Image
import cv2
from ldm_patched.utils import path_utils


def _find_upscalers():
    try:
        models = path_utils.get_filename_list("upscale_models")
    except Exception as e:
        print(f"Failed to load upscale models: {e}")
        models = []
    return ['None'] + models


def _resample_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize mask array to the given width and height using linear interpolation."""
    return cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)


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
    tile_image: Optional[Image.Image] = None  # Store the actual tile for img2img processing


class OptimizedUpscaler:
    """Fooocus optimized upscaler with smart tiling, caching, and denoising support."""

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
        """Simplified tile calculation similar to A1111 approach."""
        tiles = []
        
        # Calculate grid dimensions
        cols = max(math.ceil((width - overlap) / (tile_size - overlap)), 1)
        rows = max(math.ceil((height - overlap) / (tile_size - overlap)), 1)
        
        print(f"[FooocusUpscale] Grid: {rows}x{cols}, tile_size: {tile_size}, overlap: {overlap}")
        
        # Simple grid calculation like A1111
        for row in range(rows):
            for col in range(cols):
                # Source coordinates (original image)
                src_x = col * (tile_size - overlap)
                src_y = row * (tile_size - overlap)
                src_w = min(tile_size, width - src_x)
                src_h = min(tile_size, height - src_y)
                
                # Destination coordinates (upscaled image)
                dst_x = int(src_x * scale_factor)
                dst_y = int(src_y * scale_factor)
                dst_w = int(src_w * scale_factor)
                dst_h = int(src_h * scale_factor)
                
                tiles.append(TileInfo(
                    src_x=src_x, src_y=src_y, src_w=src_w, src_h=src_h,
                    dst_x=dst_x, dst_y=dst_y, dst_w=dst_w, dst_h=dst_h,
                    overlap_mask=None  # No complex masks, simple placement
                ))
        return tiles

    def _create_overlap_mask(self, w: int, h: int, overlap: int,
                            row: int, col: int, rows: int, cols: int) -> np.ndarray:
        mask = np.ones((h, w), dtype=np.float32)
        fade_size = min(overlap, w // 4, h // 4)
        
        if fade_size > 0:
            if col > 0:  # Not leftmost column
                for i in range(fade_size):
                    mask[:, i] *= i / fade_size
            if col < cols - 1:  # Not rightmost column
                for i in range(fade_size):
                    mask[:, w - 1 - i] *= i / fade_size
            if row > 0:  # Not topmost row
                for i in range(fade_size):
                    mask[i, :] *= i / fade_size
            if row < rows - 1:  # Not bottommost row
                for i in range(fade_size):
                    mask[h - 1 - i, :] *= i / fade_size
        return mask

    def _get_uov_method(self):
        """Get the correct upscale method reference."""
        try:
            from modules import flags
            # Try different possible attribute names for upscale method
            if hasattr(flags, 'uov_method_vary'):
                return flags.uov_method_vary
            elif hasattr(flags, 'vary'):
                return flags.vary
            elif hasattr(flags, 'upscale_vary'):
                return flags.upscale_vary
            else:
                # Fallback to string if no flag found
                return 'vary'
        except Exception as e:
            print(f"[FooocusUpscale] Could not get uov_method from flags: {e}")
            return 'vary'

    def _get_performance_value(self):
        """Get the correct performance value as string."""
        try:
            from modules import flags
            # Try to get the string value instead of the enum object
            if hasattr(flags.Performance, 'QUALITY'):
                # If it's an enum, get its value
                quality_enum = flags.Performance.QUALITY
                if hasattr(quality_enum, 'value'):
                    return quality_enum.value
                elif hasattr(quality_enum, 'name'):
                    return quality_enum.name
                else:
                    return str(quality_enum)
            else:
                # Fallback to common Fooocus performance strings
                return 'Quality'
        except Exception as e:
            print(f"[FooocusUpscale] Could not get performance from flags: {e}")
            return 'Quality'

    def _process_tile_with_denoising(self, tile_image: Image.Image, tile_info: TileInfo,
                                   upscaler_name: str, prompt: str, denoising_strength: float) -> np.ndarray:
        """Process a single tile with optional denoising using Fooocus's vary/upscale pipeline."""
        
        # First upscale the tile if upscaler is specified
        if upscaler_name != "None":
            model = self._load_model_cached(upscaler_name)
            if model is not None:
                try:
                    tile_np = np.array(tile_image)
                    from modules.upscaler import upscale_with_model
                    upscaled_np = upscale_with_model(model, tile_np, self._device)
                    upscaled_image = Image.fromarray(upscaled_np)
                except Exception as e:
                    print(f"[FooocusUpscale] Upscaler failed: {e}")
                    upscaled_image = tile_image.resize((tile_info.dst_w, tile_info.dst_h), Image.Resampling.LANCZOS)
            else:
                upscaled_image = tile_image.resize((tile_info.dst_w, tile_info.dst_h), Image.Resampling.LANCZOS)
        else:
            upscaled_image = tile_image.resize((tile_info.dst_w, tile_info.dst_h), Image.Resampling.LANCZOS)
        
        # Apply denoising if strength > 0
        if denoising_strength > 0.0 and prompt:
            try:
                # Try to use Fooocus's async worker system with corrected parameters
                from modules.async_worker import AsyncTask
                
                # Get the correct values as strings
                uov_method = self._get_uov_method()
                performance_value = self._get_performance_value()
                
                # Create a task for this tile using the vary/upscale function
                task = AsyncTask(args=[
                    prompt,                    # prompt
                    '',                       # negative_prompt  
                    [],                       # style_selections
                    performance_value,        # performance_selection - FIXED: use string value
                    'custom',                 # aspect_ratios_selection
                    1,                        # image_number
                    -1,                       # image_seed
                    2.0,                      # sharpness
                    7.0,                      # guidance_scale
                    None,                     # base_model_name (use default)
                    None,                     # refiner_model_name (use default)
                    0.8,                      # refiner_switch
                    [],                       # loras
                    True,                     # input_image_checkbox
                    'uov',                    # current_tab
                    uov_method,               # uov_method
                    upscaled_image,           # uov_input_image
                    [],                       # outpaint_selections
                    None,                     # inpaint_input_image
                    '',                       # inpaint_additional_prompt
                    None,                     # inpaint_mask_image_upload
                    True,                     # disable_preview
                    True,                     # disable_intermediate_results
                    True,                     # disable_seed_increment
                    1.5,                      # adm_scaler_positive
                    0.8,                      # adm_scaler_negative
                    0.3,                      # adm_scaler_end
                    7.0,                      # adaptive_cfg
                    2,                        # clip_skip
                    'dpmpp_2m',              # sampler_name
                    'karras',                # scheduler_name
                    'Default (model)',       # vae_name
                    -1,                      # overwrite_step
                    -1,                      # overwrite_switch
                    tile_info.dst_w,         # overwrite_width
                    tile_info.dst_h,         # overwrite_height
                    denoising_strength,      # overwrite_vary_strength
                    -1,                      # overwrite_upscale_strength
                    False,                   # mixing_image_prompt_and_vary_upscale
                    False,                   # mixing_image_prompt_and_inpaint
                    False,                   # debugging_cn_preprocessor
                    False,                   # skipping_cn_preprocessor
                    0.25,                    # controlnet_softness
                    64,                      # canny_low_threshold
                    128,                     # canny_high_threshold
                    False,                   # freeu_enabled
                    1.01,                    # freeu_b1
                    1.02,                    # freeu_b2
                    0.99,                    # freeu_s1
                    0.95,                    # freeu_s2
                    False,                   # debugging_inpaint_preprocessor
                    False,                   # inpaint_disable_initial_latent
                    'v1',                    # inpaint_engine
                    1.0,                     # inpaint_strength
                    1.0,                     # inpaint_respective_field
                    False,                   # invert_mask_checkbox
                    0,                       # inpaint_erode_or_dilate
                ])
                
                # Process the task synchronously
                results = task.execute()
                
                if results and len(results) > 0:
                    denoised_image = results[0]
                    return np.array(denoised_image, dtype=np.float32)
                    
            except Exception as e:
                print(f"[FooocusUpscale] Denoising failed for tile: {e}")
                print(f"[FooocusUpscale] Falling back to upscaled image without denoising")
        
        # Return upscaled image as numpy array
        result_np = np.array(upscaled_image, dtype=np.float32)
        return result_np

    def _process_tile_batch_with_denoising(self, tiles_data: List[Tuple[Image.Image, TileInfo]],
                                          upscaler_name: str, prompt: str, 
                                          denoising_strength: float) -> List[Tuple[np.ndarray, TileInfo]]:
        """Process batch of tiles with upscaling and optional denoising."""
        results = []
        
        for tile_image, info in tiles_data:
            try:
                processed_tile = self._process_tile_with_denoising(
                    tile_image, info, upscaler_name, prompt, denoising_strength
                )
                results.append((processed_tile, info))
            except Exception as e:
                print(f"[FooocusUpscale] Tile processing failed: {e}")
                # Fallback to simple resize
                fallback_np = np.array(tile_image.resize((info.dst_w, info.dst_h), Image.Resampling.LANCZOS), dtype=np.float32)
                results.append((fallback_np, info))
        
        return results

    def _blend_tile_simple(self, canvas: np.ndarray, tile: np.ndarray, tile_info: TileInfo):
        """Simple tile placement similar to A1111/Forge approach - no complex blending."""
        dst_x, dst_y = tile_info.dst_x, tile_info.dst_y
        dst_h, dst_w = tile.shape[:2]
        
        # Ensure boundaries
        end_y = min(dst_y + dst_h, canvas.shape[0])
        end_x = min(dst_x + dst_w, canvas.shape[1])
        actual_h = end_y - dst_y
        actual_w = end_x - dst_x
        
        if actual_h <= 0 or actual_w <= 0:
            return
        
        # Simple direct placement - just like the original A1111 approach
        canvas[dst_y:end_y, dst_x:end_x] = tile[:actual_h, :actual_w]

    def upscale_image(self, image: Image.Image, overlap: int, scale_factor: float,
                     tile_size: int = 512, upscaler_name: str = "None",
                     progress_callback: Optional[Callable] = None,
                     batch_size: int = 4, prompt: str = "", 
                     denoising_strength: float = 0.0) -> Image.Image:
        w, h = image.size
        final_w = int(w * scale_factor)
        final_h = int(h * scale_factor)
        
        print(f"[FooocusUpscale] {w}x{h} -> {final_w}x{final_h} (scale: {scale_factor}, model: {upscaler_name})")
        if denoising_strength > 0:
            print(f"[FooocusUpscale] Denoising strength: {denoising_strength}, Prompt: '{prompt[:50]}...'")
        
        tiles = self._calculate_smart_tiles(w, h, tile_size, overlap, scale_factor)
        total_tiles = len(tiles)
        
        # Initialize canvas - simple approach like A1111
        canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        
        processed_count = 0
        
        # Process tiles with denoising support
        for batch_start in range(0, total_tiles, batch_size):
            batch_end = min(batch_start + batch_size, total_tiles)
            batch_tiles = tiles[batch_start:batch_end]
            batch_data = []
            
            for info in batch_tiles:
                # Extract tile from original image
                tile_image = image.crop((
                    info.src_x, info.src_y,
                    info.src_x + info.src_w, info.src_y + info.src_h
                ))
                batch_data.append((tile_image, info))
            
            try:
                # Process batch with denoising
                processed_batch = self._process_tile_batch_with_denoising(
                    batch_data, upscaler_name, prompt, denoising_strength
                )
                
                for processed_tile, info in processed_batch:
                    # Ensure processed tile is in correct format
                    if processed_tile.dtype != np.uint8:
                        if processed_tile.max() <= 1.0:
                            processed_tile = (processed_tile * 255.0).astype(np.uint8)
                        else:
                            processed_tile = np.clip(processed_tile, 0, 255).astype(np.uint8)
                    
                    # Simple placement like A1111 - no complex blending
                    self._blend_tile_simple(canvas, processed_tile, info)
                    
                    processed_count += 1
                    if progress_callback:
                        preview = None
                        if processed_count % 5 == 0:
                            preview = Image.fromarray(canvas)
                        progress_callback(processed_count, total_tiles, preview)
                        
            except Exception as e:
                print(f"[FooocusUpscale] Batch processing error: {e}")
                import traceback
                traceback.print_exc()
                processed_count += len(batch_tiles)
                if progress_callback:
                    progress_callback(processed_count, total_tiles, None)
            
            # Memory cleanup every few batches
            if batch_start % (batch_size * 3) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Debug: Check if canvas is all black
        if canvas.max() == 0:
            print("[FooocusUpscale] WARNING: Final canvas is all black!")
            
            # Fallback: simple resize if processing failed
            print("[FooocusUpscale] Falling back to simple resize...")
            return image.resize((final_w, final_h), Image.Resampling.LANCZOS)
        
        result_image = Image.fromarray(canvas)
        
        # Cleanup
        del canvas
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
        batch_size=4,
        prompt=prompt,
        denoising_strength=denoising_strength,
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
