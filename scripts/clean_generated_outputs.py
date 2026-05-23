from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_PATTERNS = [
    "outputs",
    "reports",
    "reports_iou*",
    "demo_assets",
    "tracklet_reports*",
    "tracking_reports*",
    "tracklet_replay*",
    "tracklet_analysis*",
    "tracklet_curation*",
    "tracklet_demo_export*",
    "showcase_export*",
]


def discover_generated_paths(project_root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []

    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path.exists():
                paths.append(path)

    return sorted(set(paths), key=lambda p: str(p).lower())


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove generated AutoPercept3D outputs while keeping source code and dataset untouched."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root. Default: current directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be removed.",
    )
    parser.add_argument(
        "--include-media",
        action="store_true",
        help="Also remove generated GIF/MP4/AVI/MOV files directly under the project root.",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    patterns = list(DEFAULT_PATTERNS)

    if args.include_media:
        patterns.extend(["*.gif", "*.mp4", "*.avi", "*.mov"])

    paths = discover_generated_paths(project_root, patterns)

    print("=" * 60)
    print("AutoPercept3D generated-output cleanup")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Dry run:      {args.dry_run}")
    print()

    if not paths:
        print("No generated output paths found.")
        return

    print("Generated paths:")
    for path in paths:
        kind = "dir " if path.is_dir() else "file"
        print(f"  [{kind}] {path.relative_to(project_root)}")

    if args.dry_run:
        print()
        print("Dry run only. No files were removed.")
        return

    print()
    print("Removing generated paths...")

    for path in paths:
        remove_path(path)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
