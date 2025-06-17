# Simple caching utilities to replace missing dependency from upstream.
# This module provides minimal functionality required by other modules
# such as NewLoraSystem.sd_forge_lora.network.

import os
from typing import Callable, Any, Dict

# cache data structured as: {subsection: {title: {"mtime": float, "value": Any}}}
_cache_store: Dict[str, Dict[str, Dict[str, Any]]] = {}


def cached_data_for_file(subsection: str, title: str, filename: str,
                         func: Callable[[], Any]):
    """Return cached data for *filename* or generate it using *func*.

    Data is cached in-memory only. The cache key is a combination of
    *subsection* and *title*. The on-disk modification time of
    *filename* is used to detect changes and invalidate the cached
    value.
    """
    subsection_store = _cache_store.setdefault(subsection, {})
    entry = subsection_store.get(title)
    ondisk_mtime = os.path.getmtime(filename)

    if entry is None or entry.get("mtime") < ondisk_mtime:
        value = func()
        if value is None:
            return None
        entry = {"mtime": ondisk_mtime, "value": value}
        subsection_store[title] = entry

    return entry["value"]
