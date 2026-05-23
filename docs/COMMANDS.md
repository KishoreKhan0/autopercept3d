# AutoPercept3D Command Reference

This file is a project command checklist. It is intentionally separate from the README so the README can stay clean and portfolio-focused later.

## 1. Activate environment

```bat
cd C:\Users\KISHORE KHAN\Desktop\Projects\autopercept3d
.venv\Scripts\activate
```

## 2. Install dependencies

```bat
pip install -r requirements.txt
```

## 3. Set dataset root

```bat
set KITTI_ROOT=C:\Users\KISHORE KHAN\Desktop\Projects\datasets\kitti_object
```

## 4. Inspect one KITTI frame

```bat
python scripts\inspect_kitti_frame.py --dataset-root "%KITTI_ROOT%" --frame 000000
```

## 5. Run single-frame perception pipeline

```bat
python scripts\run_frame_pipeline.py --dataset-root "%KITTI_ROOT%" --frame 000264 --output-dir outputs
```

## 6. Render main frame demo assets

```bat
python scripts\export_demo_assets.py --dataset-root "%KITTI_ROOT%" --frame 000264 --output-dir demo_assets
```

## 7. Evaluate one frame with lightweight BEV IoU

```bat
python scripts\evaluate_frame_iou.py --dataset-root "%KITTI_ROOT%" --frame 000264 --iou-threshold 0.10 --output-json reports\eval_000264.json --save-plot reports\eval_000264.png
```

## 8. Run batch BEV IoU evaluation

```bat
python scripts\run_batch_iou_evaluation.py --dataset-root "%KITTI_ROOT%" --start-index 250 --max-frames 20 --iou-threshold 0.10 --output-dir reports_iou_250_269
```

## 9. Export stable tracklet replay

```bat
python scripts\render_tracklet_replay.py --dataset-root "%KITTI_ROOT%" --start-index 250 --max-frames 20 --box-type oriented --output-dir tracklet_replay_250_269
```

## 10. Analyze stable tracklets

```bat
python scripts\analyze_tracklets.py --tracklets-csv tracklet_replay_250_269\stable_tracklets.csv --output-dir tracklet_analysis_250_269
```

## 11. Curate high-quality tracklets

```bat
python scripts\curate_tracklets.py --tracklets-csv tracklet_replay_250_269\stable_tracklets.csv --output-dir tracklet_curation_250_269
```

## 12. Export tracklet GIF/contact sheet

```bat
python scripts\export_tracklet_gif.py --frames-dir tracklet_replay_250_269\frames --output-dir tracklet_demo_export_250_269 --duration-ms 700
```

## 13. Run main dashboard

```bat
streamlit run scripts\dashboard_app.py
```

## 14. Run tracklet review dashboard

```bat
streamlit run scripts\dashboard_tracklets_app.py
```

## 15. One-command showcase export

```bat
python scripts\export_showcase_package.py --dataset-root "%KITTI_ROOT%" --frame 000264 --start-index 250 --max-frames 20 --output-dir showcase_export_250_269
```

## 16. Optional cleanup of generated outputs

Preview what would be removed:

```bat
python scripts\clean_generated_outputs.py --dry-run
```

Actually remove generated folders:

```bat
python scripts\clean_generated_outputs.py
```

## Notes

- The KITTI dataset should stay outside Git and should not be committed.
- Generated reports, GIFs, replay frames, and showcase exports are ignored by `.gitignore`.
- BEV IoU is a lightweight internal proposal-quality metric, not official KITTI evaluation.
- Keep one or two selected small demo images only if you intentionally want them in GitHub.
