from importlib import import_module
import sys

sys.modules.setdefault("adetailer", sys.modules[__name__])
for sub in ("args", "common", "mediapipe", "ultralytics"):
    sys.modules[f"adetailer.{sub}"] = sys.modules[__name__]

_base = import_module("modules.adetailer.vendor_adetailer")

for sub in ("args", "common", "mediapipe", "ultralytics"):
    sys.modules[f"adetailer.{sub}"] = import_module(f"modules.adetailer.vendor_adetailer.{sub}")

__all__ = list(getattr(_base, "__all__", []))
globals().update({name: getattr(_base, name) for name in __all__})
