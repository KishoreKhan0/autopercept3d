from __future__ import annotations

import argparse
import json
from pathlib import Path

from autopercept3d.visualization.sequence_export import (
    discover_frame_images,
    export_gif_from_images,
    make_contact_sheet,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an animated GIF and contact sheet from AutoPercept3D tracklet replay frames."
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing rendered replay frame images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tracklet_demo_export",
        help="Output directory for GIF/contact sheet.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Image filename pattern inside frames-dir.",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=450,
        help="Duration per GIF frame in milliseconds.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1200,
        help="Maximum GIF width. Use 0 to keep original width.",
    )
    parser.add_argument(
        "--contact-sheet-columns",
        type=int,
        default=4,
        help="Number of columns in the contact sheet.",
    )

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_width = args.max_width if args.max_width > 0 else None

    print("=" * 60)
    print("AutoPercept3D tracklet GIF export")
    print("=" * 60)
    print(f"Frames directory:     {frames_dir}")
    print(f"Output directory:     {output_dir}")
    print(f"Pattern:              {args.pattern}")
    print(f"Duration per frame:   {args.duration_ms} ms")
    print(f"Max GIF width:        {max_width if max_width else 'original'}")
    print()

    image_paths = discover_frame_images(frames_dir, pattern=args.pattern)

    output_gif = output_dir / "tracklet_replay.gif"
    contact_sheet = output_dir / "tracklet_replay_contact_sheet.png"
    summary_json = output_dir / "tracklet_demo_export_summary.json"

    export_gif_from_images(
        image_paths=image_paths,
        output_gif=output_gif,
        duration_ms=args.duration_ms,
        max_width=max_width,
    )

    make_contact_sheet(
        image_paths=image_paths,
        output_path=contact_sheet,
        columns=args.contact_sheet_columns,
    )

    summary = {
        "frames_dir": str(frames_dir),
        "num_frames": len(image_paths),
        "first_frame_image": str(image_paths[0]),
        "last_frame_image": str(image_paths[-1]),
        "duration_ms": args.duration_ms,
        "max_width": max_width,
        "output_gif": str(output_gif),
        "contact_sheet": str(contact_sheet),
    }

    write_json(summary_json, summary)

    print("=" * 60)
    print("GIF export summary")
    print("=" * 60)
    print(f"Frames used:          {len(image_paths)}")
    print(f"First frame:          {image_paths[0].name}")
    print(f"Last frame:           {image_paths[-1].name}")
    print()
    print("Saved outputs:")
    print(f"  GIF:                {output_gif}")
    print(f"  Contact sheet:      {contact_sheet}")
    print(f"  Summary JSON:       {summary_json}")


if __name__ == "__main__":
    main()
