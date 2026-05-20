from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from autopercept3d.datasets.kitti import KITTIDataset
from autopercept3d.evaluation.matching import evaluate_bev_proposals, evaluation_to_dict
from autopercept3d.perception.pipeline import ClassicalPipelineConfig, run_classical_lidar_pipeline
from autopercept3d.visualization.bev import (
    plot_cluster_points_bev,
    plot_ground_truth_boxes_bev,
    plot_points_bev,
    plot_proposal_boxes_bev,
    set_bev_axes,
)
from autopercept3d.visualization.camera import draw_camera_panel


DEFAULT_DATASET_ROOT = os.environ.get(
    "KITTI_ROOT",
    r"<PATH_TO_KITTI_OBJECT_DATASET>",
)


def list_frame_ids(dataset_root: str, max_frames: int = 300) -> list[str]:
    velodyne_dir = Path(dataset_root) / "training" / "velodyne"

    if not velodyne_dir.exists():
        return []

    frame_ids = sorted(path.stem for path in velodyne_dir.glob("*.bin"))
    return frame_ids[:max_frames]


def make_camera_figure(image, labels, frame_id: str):
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()

    draw_camera_panel(
        ax,
        image=image,
        labels=labels,
        title=f"Camera image - frame {frame_id}",
    )

    plt.tight_layout()
    return fig


def make_bev_summary_figure(result, labels, calibration):
    fig = plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plot_points_bev(
        result.ground_result.ground_points,
        label="ground",
        point_size=1.0,
        alpha=0.25,
    )
    plot_points_bev(
        result.ground_result.non_ground_points,
        label="non-ground",
        point_size=1.0,
        alpha=0.75,
    )
    set_bev_axes(f"Ground removal - frame {result.frame_id}")
    plt.legend(markerscale=5)

    plt.subplot(1, 2, 2)
    plot_cluster_points_bev(
        result.filtered_cluster_points,
        result.filtered_cluster_labels,
    )
    plot_proposal_boxes_bev(result.boxes, show_ids=True)
    plot_ground_truth_boxes_bev(labels, calibration)

    title = (
        f"Object proposals - frame {result.frame_id} | "
        f"boxes={result.metrics['boxes']} | "
        f"fps={result.metrics['approx_fps']:.2f}"
    )
    set_bev_axes(title)
    plt.legend(markerscale=5)

    plt.tight_layout()
    return fig


def make_timing_figure(timings_ms: dict):
    stage_names = [key for key in timings_ms.keys() if key != "total"]
    values = [timings_ms[key] for key in stage_names]

    fig = plt.figure(figsize=(9, 4))
    plt.barh(stage_names, values)
    plt.xlabel("Time [ms]")
    plt.title("Pipeline stage timings")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def metric_box(label: str, value):
    if isinstance(value, float):
        st.metric(label, f"{value:.3f}" if value < 1.0 else f"{value:.2f}")
    else:
        st.metric(label, str(value))


def main() -> None:
    st.set_page_config(
        page_title="AutoPercept3D Dashboard + IoU",
        page_icon="🚗",
        layout="wide",
    )

    st.title("AutoPercept3D")
    st.caption("KITTI LiDAR perception, visualization, and BEV IoU evaluation workbench")

    st.sidebar.header("Dataset")

    dataset_root = st.sidebar.text_input(
        "KITTI dataset root",
        value=DEFAULT_DATASET_ROOT,
    )

    frame_limit = st.sidebar.slider(
        "Frames to list",
        min_value=10,
        max_value=1000,
        value=300,
        step=10,
    )

    frame_ids = list_frame_ids(dataset_root, max_frames=frame_limit)

    if not frame_ids:
        st.error(
            "No KITTI frames found. Check that the dataset root contains "
            "`training\\velodyne`."
        )
        st.stop()

    selected_frame = st.sidebar.selectbox(
        "Frame ID",
        options=frame_ids,
        index=0,
    )

    st.sidebar.header("Pipeline parameters")

    voxel_size = st.sidebar.slider(
        "Voxel size [m]",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
    )

    ground_threshold = st.sidebar.slider(
        "Ground distance threshold [m]",
        min_value=0.05,
        max_value=0.60,
        value=0.25,
        step=0.05,
    )

    ground_iterations = st.sidebar.slider(
        "Ground RANSAC iterations",
        min_value=30,
        max_value=300,
        value=120,
        step=10,
    )

    dbscan_eps = st.sidebar.slider(
        "DBSCAN eps [m]",
        min_value=0.20,
        max_value=2.00,
        value=0.80,
        step=0.05,
    )

    dbscan_min_samples = st.sidebar.slider(
        "DBSCAN min_samples",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )

    st.sidebar.header("BEV IoU evaluation")

    iou_threshold = st.sidebar.slider(
        "BEV IoU match threshold",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        help="Lightweight AABB BEV IoU threshold. This is not official KITTI evaluation.",
    )

    run_clicked = st.sidebar.button("Run pipeline", type="primary")

    if not run_clicked:
        st.info("Choose a frame and click **Run pipeline**.")
        st.stop()

    config = ClassicalPipelineConfig(
        voxel_size=voxel_size,
        ground_distance_threshold=ground_threshold,
        ground_iterations=ground_iterations,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    with st.spinner(f"Running pipeline and IoU evaluation for frame {selected_frame}..."):
        dataset = KITTIDataset(dataset_root=dataset_root)

        image = dataset.load_image(selected_frame)
        labels = dataset.load_labels(selected_frame)
        calibration = dataset.load_calibration(selected_frame)

        result = run_classical_lidar_pipeline(
            dataset=dataset,
            frame_id=selected_frame,
            config=config,
        )

        eval_result = evaluate_bev_proposals(
            proposal_boxes_3d=result.boxes,
            raw_labels=labels,
            calibration=calibration,
            iou_threshold=iou_threshold,
        )
        eval_payload = evaluation_to_dict(eval_result)

    st.success(f"Pipeline finished for frame {selected_frame}")

    st.subheader("Frame metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_box("Raw points", result.metrics["raw_points"])
    with col2:
        metric_box("Downsampled", result.metrics["downsampled_points"])
    with col3:
        metric_box("Non-ground", result.metrics["non_ground_points"])
    with col4:
        metric_box("Boxes", result.metrics["boxes"])
    with col5:
        metric_box("Approx FPS", result.metrics["approx_fps"])

    st.subheader("BEV IoU evaluation summary")

    e1, e2, e3, e4, e5 = st.columns(5)

    with e1:
        metric_box("GT objects", eval_payload["num_ground_truth"])
    with e2:
        metric_box("Matches", eval_payload["num_matches"])
    with e3:
        metric_box("Precision", eval_payload["precision"])
    with e4:
        metric_box("Recall", eval_payload["recall"])
    with e5:
        metric_box("Mean IoU", eval_payload["mean_iou"])

    tab_camera, tab_bev, tab_iou, tab_metrics = st.tabs(
        ["Camera", "BEV perception", "BEV IoU evaluation", "Metrics + timings"]
    )

    with tab_camera:
        st.write("KITTI camera image with 2D ground-truth labels.")
        camera_fig = make_camera_figure(image, labels, selected_frame)
        st.pyplot(camera_fig, clear_figure=True)

    with tab_bev:
        st.write(
            "Left: ground vs non-ground. Right: clusters, proposal boxes, "
            "and KITTI ground-truth boxes."
        )
        bev_fig = make_bev_summary_figure(result, labels, calibration)
        st.pyplot(bev_fig, clear_figure=True)

    with tab_iou:
        st.write(
            "Lightweight AABB BEV IoU matching between proposal boxes and KITTI labels. "
            "This is an internal debugging metric, not official KITTI evaluation."
        )

        st.json(
            {
                "iou_threshold": iou_threshold,
                "num_proposals": eval_payload["num_proposals"],
                "num_ground_truth": eval_payload["num_ground_truth"],
                "num_matches": eval_payload["num_matches"],
                "precision": eval_payload["precision"],
                "recall": eval_payload["recall"],
                "mean_iou": eval_payload["mean_iou"],
            }
        )

        if eval_payload["matches"]:
            st.write("Matched proposal / ground-truth pairs")
            st.dataframe(eval_payload["matches"], use_container_width=True)
        else:
            st.warning("No matches found at this IoU threshold.")

        with st.expander("Unmatched IDs"):
            st.write("Unmatched proposal IDs")
            st.write(eval_payload["unmatched_proposal_ids"])
            st.write("Unmatched ground-truth IDs")
            st.write(eval_payload["unmatched_gt_ids"])

    with tab_metrics:
        left, right = st.columns([1, 1])

        with left:
            st.write("Pipeline metrics")
            st.json(result.metrics)

        with right:
            st.write("Timing breakdown")
            timing_fig = make_timing_figure(result.timings_ms)
            st.pyplot(timing_fig, clear_figure=True)

            st.write("Raw timings")
            st.json(result.timings_ms)


if __name__ == "__main__":
    main()
