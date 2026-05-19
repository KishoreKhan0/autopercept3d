@echo off
call .venv\Scripts\activate.bat
python scripts\inspect_kitti_frame.py --dataset-root C:\datasets\kitti_object --frame 000000
