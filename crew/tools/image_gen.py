# FILE: crew/tools/image_gen.py
# Purpose: Save a 1x1 PNG placeholder so the pipeline works without image APIs.

from pathlib import Path

def save_placeholder_image(path="artifacts/image.png"):
    """
    Writes a 1x1 transparent PNG to artifacts/image.png and returns the path.
    Replace with a real image-generation call later if you want.
    """
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    # tiny transparent PNG
    with open(path, "wb") as f:
        f.write(bytes.fromhex(
          "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
          "0000000A49444154789C6360000002000154A24F5D0000000049454E44AE426082"
        ))
    return path
