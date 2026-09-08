"""Import the named GitHub ZIPs into lightweight, transparent UI assets.

Usage: python scripts/prepare_robot_assets.py /path/to/downloaded/archives
Requires Pillow and NumPy. Original archives are not modified or committed twice.
Only explicitly allowlisted filenames are read (no extractall/path traversal).
"""

from collections import deque
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image, ImageFilter

FAMILIES = {
    "calendar": ["normal_robot", "detecting_intent", "clock_face", "mode", "confirmed"],
    "reminder": [
        "normal_robot",
        "detecting_intent",
        "notepad_appears",
        "writing_task",
        "saved",
    ],
    "summarizer": [
        "normal_robot",
        "thinking_begins",
        "deep_processing",
        "information_compression",
        "summary_ready",
    ],
}
MODES = [
    "default_mode_robot_blue",
    "privacy_mode_robot_green",
    "global_model_robot_soft_red",
]
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend/public/brand/robots"


def remove_checkerboard(image):
    """Remove only edge-connected light neutral backdrop; preserve enclosed metal.

    The supplied frame PNGs have an opaque checkerboard baked into their RGB pixels,
    not actual transparency. No colour-keying is applied inside the robot itself.
    """
    rgba = np.array(image.convert("RGBA"))
    rgb = rgba[:, :, :3].astype(int)
    eligible = (rgb.min(axis=2) >= 185) & ((rgb.max(axis=2) - rgb.min(axis=2)) <= 23)
    height, width = eligible.shape
    seen = np.zeros((height, width), dtype=bool)
    queue = deque()
    for x in range(width):
        for y in (0, height - 1):
            if eligible[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if eligible[y, x] and not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= ny < height
                and 0 <= nx < width
                and eligible[ny, nx]
                and not seen[ny, nx]
            ):
                seen[ny, nx] = True
                queue.append((ny, nx))
    rgba[seen, 3] = 0
    # Remove narrow neighbouring-panel slivers already cut off at the source edge.
    rgba[:, :2, 3] = 0
    rgba[:, -2:, 3] = 0
    result = Image.fromarray(rgba)
    result.putalpha(result.getchannel("A").filter(ImageFilter.GaussianBlur(0.35)))
    return result


def main(source, revision):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_repository": "code-bhaskar/Privacy-Preserving-Digital-Assistant",
        "source_revision": revision,
        "files": [],
    }
    frames = {}
    for archive, names in [
        (
            "robot_gif_frames_named.zip",
            [
                f"{family}_{suffix}"
                for family, suffixes in FAMILIES.items()
                for suffix in suffixes
            ],
        ),
        ("robot_modes_separate_named.zip", MODES),
    ]:
        data = (source / archive).read_bytes()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for stem in names:
                raw = z.read(stem + ".png")
                image = Image.open(io.BytesIO(raw)).convert("RGBA")
                if archive == "robot_gif_frames_named.zip":
                    image = remove_checkerboard(image)
                    if stem.endswith("normal_robot") or stem in (
                        "reminder_notepad_appears",
                        "summarizer_thinking_begins",
                    ):
                        alpha = np.array(image.getchannel("A"))
                        alpha[:, -14:] = (
                            0  # Crop-neighbour slivers, not part of this character/frame.
                        )
                        image.putalpha(Image.fromarray(alpha))
                    frames[stem] = image
                else:
                    image.thumbnail((480, 480), Image.Resampling.LANCZOS)
                image.save(OUT / (stem + ".webp"), "WEBP", quality=90, method=4)
                manifest["files"].append(
                    {
                        "file": stem + ".webp",
                        "archive": archive,
                        "original": stem + ".png",
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "archive_sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
    for family, suffixes in FAMILIES.items():
        sequence = []
        for suffix in suffixes + [suffixes[0]]:
            image = frames[f"{family}_{suffix}"]
            # GIF demos use a plain dark background, unlike transparent runtime WebP frames.
            canvas = Image.new("RGBA", image.size, "#171b24")
            canvas.alpha_composite(image)
            sequence.append(canvas.convert("RGB"))
        sequence[0].save(
            OUT / (family + "-sequence.gif"),
            save_all=True,
            append_images=sequence[1:],
            duration=[600, 450, 550, 650, 1000, 600],
            loop=0,
            disposal=2,
        )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Imported {len(manifest['files'])} named images and exported 3 reference GIF sequences."
    )


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        sys.argv[2] if len(sys.argv) > 2 else "local archives; revision unspecified",
    )
