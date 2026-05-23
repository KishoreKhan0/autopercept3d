from __future__ import annotations

from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


FRAME_ID_PATTERN = re.compile(r"(\d{6})")


def discover_frame_images(frames_dir: Path, pattern: str = "*.png") -> list[Path]:
    """
    Return frame images sorted by the 6-digit KITTI frame ID in the filename.
    """
    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")

    images = sorted(frames_dir.glob(pattern))

    if not images:
        raise FileNotFoundError(f"No frame images matching '{pattern}' found in {frames_dir}")

    def sort_key(path: Path):
        match = FRAME_ID_PATTERN.search(path.stem)
        if match:
            return int(match.group(1))
        return path.name

    return sorted(images, key=sort_key)


def resize_keep_aspect(image: Image.Image, max_width: int | None = None) -> Image.Image:
    """
    Resize an image to max_width while keeping aspect ratio.
    """
    if max_width is None or max_width <= 0:
        return image.copy()

    width, height = image.size

    if width <= max_width:
        return image.copy()

    scale = max_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def normalize_frame_sizes(frames: list[Image.Image]) -> list[Image.Image]:
    """
    GIF frames must have consistent sizes. Pad smaller frames onto a white canvas.
    """
    if not frames:
        return []

    max_width = max(frame.size[0] for frame in frames)
    max_height = max(frame.size[1] for frame in frames)

    normalized: list[Image.Image] = []

    for frame in frames:
        if frame.mode != "RGB":
            frame = frame.convert("RGB")

        canvas = Image.new("RGB", (max_width, max_height), "white")
        offset_x = (max_width - frame.size[0]) // 2
        offset_y = (max_height - frame.size[1]) // 2
        canvas.paste(frame, (offset_x, offset_y))
        normalized.append(canvas)

    return normalized


def export_gif_from_images(
    image_paths: list[Path],
    output_gif: Path,
    duration_ms: int = 450,
    max_width: int | None = 1200,
    loop: int = 0,
) -> Path:
    """
    Export an animated GIF from ordered image paths.
    """
    if not image_paths:
        raise ValueError("No image paths were provided.")

    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            frame = image.convert("RGB")
            frame = resize_keep_aspect(frame, max_width=max_width)
            frames.append(frame)

    frames = normalize_frame_sizes(frames)

    first, rest = frames[0], frames[1:]

    first.save(
        output_gif,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )

    return output_gif


def make_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    columns: int = 4,
    thumb_width: int = 420,
    padding: int = 20,
    label_height: int = 34,
) -> Path:
    """
    Create a contact sheet for quick visual review of replay frames.
    """
    if not image_paths:
        raise ValueError("No image paths were provided.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thumbnails: list[tuple[Image.Image, str]] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = resize_keep_aspect(image, max_width=thumb_width)

            match = FRAME_ID_PATTERN.search(image_path.stem)
            label = match.group(1) if match else image_path.stem
            thumbnails.append((image, label))

    columns = max(1, int(columns))
    rows = (len(thumbnails) + columns - 1) // columns

    cell_width = thumb_width + padding * 2
    thumb_height = max(image.size[1] for image, _ in thumbnails)
    cell_height = thumb_height + label_height + padding * 2

    sheet_width = columns * cell_width
    sheet_height = rows * cell_height

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (thumb, label) in enumerate(thumbnails):
        row = idx // columns
        col = idx % columns

        x0 = col * cell_width + padding
        y0 = row * cell_height + padding

        sheet.paste(thumb, (x0, y0))

        text_x = x0
        text_y = y0 + thumb_height + 8
        draw.text((text_x, text_y), f"frame {label}", fill="black")

    sheet.save(output_path)
    return output_path
