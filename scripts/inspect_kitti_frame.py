import argparse

from autopercept3d.datasets.kitti import KITTIDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one KITTI Object Detection frame."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"C:\datasets\kitti_object",
        help="Path to KITTI Object Detection dataset root.",
    )

    parser.add_argument(
        "--frame",
        type=str,
        default="000000",
        help="KITTI frame id, for example 000000.",
    )

    args = parser.parse_args()

    dataset = KITTIDataset(dataset_root=args.dataset_root)
    dataset.inspect_frame(frame_id=args.frame)


if __name__ == "__main__":
    main()
