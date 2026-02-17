# adapters/apis_safety_layer.py

from operate.models import apis
import functools


def patch_llava_recursion():
    original = apis.call_ollama_llava

    @functools.wraps(original)
    def safe_llava(*args, **kwargs):
        try:
            result = original(*args, **kwargs)
        except Exception:
            return None

        # validate result shape
        if result is not None and not isinstance(result, (list, dict)):
            raise RuntimeError("Invalid llava return type")

        return result

    apis.call_ollama_llava = safe_llava


def apply_patches():
    patch_llava_recursion()
