from __future__ import annotations

from pathlib import Path

import streamlit as st

from autopercept3d.visualization.tracklet_dashboard import (
    compute_track_metrics_from_rows,
    find_replay_frame_image,
    get_available_frame_ids,
    get_visible_history_rows,
    load_optional_csv,
    load_optional_json,
    make_rejection_bar_figure,
    make_tracklet_bev_figure,
    merge_active_track_table,
    summarize_tracklets,
)


DEFAULT_STABLE_TRACKLETS = "tracklet_replay_250_269/stable_tracklets.csv"
DEFAULT_CURATED_TRACKLETS = "tracklet_curation_250_269/curated_tracklets.csv"
DEFAULT_CURATED_METRICS = "tracklet_curation_250_269/curated_track_metrics.csv"
DEFAULT_ALL_QUALITY_METRICS = "tracklet_curation_250_269/all_track_metrics_with_quality.csv"
DEFAULT_CURATION_SUMMARY = "tracklet_curation_250_269/curation_summary.json"
DEFAULT_REPLAY_FRAMES_DIR = "tracklet_replay_250_269/frames"


def metric_box(label: str, value):
    if isinstance(value, float):
        st.metric(label, f"{value:.2f}")
    else:
        st.metric(label, str(value))


def main() -> None:
    st.set_page_config(
        page_title="AutoPercept3D Tracklet Review",
        page_icon="🚗",
        layout="wide",
    )

    st.title("AutoPercept3D Tracklet Review")
    st.caption("Stable and curated BEV tracklet inspection dashboard")

    st.sidebar.header("Tracklet report inputs")

    stable_tracklets_path = st.sidebar.text_input(
        "Stable tracklets CSV",
        value=DEFAULT_STABLE_TRACKLETS,
    )
    curated_tracklets_path = st.sidebar.text_input(
        "Curated tracklets CSV",
        value=DEFAULT_CURATED_TRACKLETS,
    )
    curated_metrics_path = st.sidebar.text_input(
        "Curated track metrics CSV",
        value=DEFAULT_CURATED_METRICS,
    )
    all_quality_metrics_path = st.sidebar.text_input(
        "All quality metrics CSV",
        value=DEFAULT_ALL_QUALITY_METRICS,
    )
    curation_summary_path = st.sidebar.text_input(
        "Curation summary JSON",
        value=DEFAULT_CURATION_SUMMARY,
    )
    replay_frames_dir = st.sidebar.text_input(
        "Replay frames directory",
        value=DEFAULT_REPLAY_FRAMES_DIR,
    )

    stable_tracklets_df = load_optional_csv(stable_tracklets_path)
    curated_tracklets_df = load_optional_csv(curated_tracklets_path)
    curated_metrics_df = load_optional_csv(curated_metrics_path)
    all_quality_metrics_df = load_optional_csv(all_quality_metrics_path)
    curation_summary = load_optional_json(curation_summary_path)

    if stable_tracklets_df is None and curated_tracklets_df is None:
        st.error("No tracklet CSV files found. Check the sidebar paths.")
        st.stop()

    if curated_tracklets_df is not None and curated_metrics_df is None:
        curated_metrics_df = compute_track_metrics_from_rows(curated_tracklets_df)

    if stable_tracklets_df is not None and all_quality_metrics_df is None:
        all_quality_metrics_df = compute_track_metrics_from_rows(stable_tracklets_df)

    frame_ids = get_available_frame_ids(stable_tracklets_df, curated_tracklets_df)
    if not frame_ids:
        st.error("No frame IDs available in the provided files.")
        st.stop()

    st.sidebar.header("Review controls")

    source_option = st.sidebar.radio(
        "Track source",
        options=["curated", "stable"],
        index=0 if curated_tracklets_df is not None else 1,
    )

    selected_frame = st.sidebar.selectbox(
        "Frame ID",
        options=frame_ids,
        index=0,
        format_func=lambda value: f"{int(value):06d}",
    )

    history_length = st.sidebar.slider(
        "History length [frames]",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
    )

    top_k_tracks = st.sidebar.slider(
        "Max visible tracks",
        min_value=5,
        max_value=40,
        value=20,
        step=1,
    )

    show_boxes = st.sidebar.checkbox("Draw oriented box footprints", value=True)

    selected_tracklets_df = curated_tracklets_df if source_option == "curated" else stable_tracklets_df
    selected_metrics_df = curated_metrics_df if source_option == "curated" else all_quality_metrics_df

    if selected_tracklets_df is None or selected_tracklets_df.empty:
        st.error(f"No rows available for source: {source_option}")
        st.stop()

    visible_rows_df = get_visible_history_rows(
        tracklets_df=selected_tracklets_df,
        selected_frame=int(selected_frame),
        history_length=history_length,
        top_k=top_k_tracks,
        ranking_metrics_df=selected_metrics_df,
    )

    active_track_table_df = merge_active_track_table(
        visible_rows_df=visible_rows_df,
        selected_frame=int(selected_frame),
        metrics_df=selected_metrics_df,
    )

    stable_summary = summarize_tracklets(stable_tracklets_df)
    curated_summary = summarize_tracklets(curated_tracklets_df)

    visible_stable_count = 0
    if stable_tracklets_df is not None and not stable_tracklets_df.empty:
        visible_stable_count = int((stable_tracklets_df["frame_id"] == int(selected_frame)).sum())

    visible_curated_count = 0
    if curated_tracklets_df is not None and not curated_tracklets_df.empty:
        visible_curated_count = int((curated_tracklets_df["frame_id"] == int(selected_frame)).sum())

    st.subheader("Tracklet summary")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_box("Visible stable tracks", visible_stable_count)
    with c2:
        metric_box("Visible curated tracks", visible_curated_count)
    with c3:
        metric_box("Total stable tracklets", stable_summary["tracks"])
    with c4:
        metric_box("Total curated tracklets", curated_summary["tracks"])
    with c5:
        if curation_summary is not None:
            metric_box("Curated ratio", float(curation_summary.get("curated_ratio", 0.0)))
        else:
            metric_box("Curated ratio", 0.0)

    tabs = st.tabs(
        [
            "BEV review",
            "Active tracks table",
            "Curation analytics",
            "Replay frame",
        ]
    )

    with tabs[0]:
        st.write(
            f"Showing **{source_option}** tracklets for frame **{int(selected_frame):06d}** "
            f"with history length **{history_length}**."
        )

        bev_fig = make_tracklet_bev_figure(
            visible_rows_df=visible_rows_df,
            selected_frame=int(selected_frame),
            title=f"{source_option.capitalize()} BEV tracklets - frame {int(selected_frame):06d}",
            show_boxes=show_boxes,
        )
        st.pyplot(bev_fig, clear_figure=True)

    with tabs[1]:
        st.write("Visible active tracks in the selected frame.")
        if active_track_table_df.empty:
            st.info("No visible active tracks for this frame.")
        else:
            st.dataframe(active_track_table_df, use_container_width=True)

        if selected_metrics_df is not None and not selected_metrics_df.empty:
            with st.expander("Full track metrics table"):
                st.dataframe(selected_metrics_df, use_container_width=True)

    with tabs[2]:
        left, right = st.columns([1, 1])

        with left:
            st.write("Curation summary")
            if curation_summary is not None:
                st.json(curation_summary)
            else:
                st.info("No curation summary JSON found.")

        with right:
            rejection_fig = make_rejection_bar_figure(curation_summary)
            st.pyplot(rejection_fig, clear_figure=True)

        st.write("Quality metrics overview")
        if all_quality_metrics_df is not None and not all_quality_metrics_df.empty:
            st.dataframe(all_quality_metrics_df, use_container_width=True)
        else:
            st.info("No quality metrics CSV found.")

    with tabs[3]:
        frame_image_path = find_replay_frame_image(replay_frames_dir, int(selected_frame))
        if frame_image_path is None:
            st.info("No replay frame image found for the selected frame.")
        else:
            st.image(str(frame_image_path), caption=frame_image_path.name, use_container_width=True)


if __name__ == "__main__":
    main()
