import os
import time
from urllib.parse import urlparse
from typing import Optional
import urllib.error


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        for attempt in range(3):
            try:
                download_url_to_file(url, cached_file, progress=progress)
                break
            except urllib.error.HTTPError as e:
                if attempt == 2:
                    raise
                print(f"Download failed with {e}. Retrying {attempt + 1}/3...")
                time.sleep(5)
    return cached_file
