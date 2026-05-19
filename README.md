# AutoPercept3D - Step 1

This is the first minimal implementation step.

It contains:
- a minimal KITTI dataset loader
- a script to inspect one KITTI frame

Expected KITTI dataset location on Windows:

```text
C:\datasets\kitti_object
```

Expected KITTI structure:

```text
C:\datasets\kitti_object\training\image_2
C:\datasets\kitti_object\training\velodyne
C:\datasets\kitti_object\training\calib
C:\datasets\kitti_object\training\label_2
```

## Windows CMD setup

From inside the project folder:

```bat
python --version
py --version
```

Use whichever works.

Create virtual environment:

```bat
py -m venv .venv
```

Activate:

```bat
.venv\Scripts\activate.bat
```

Install the project:

```bat
python -m pip install --upgrade pip
pip install -e .
```

Run the inspector:

```bat
python scripts\inspect_kitti_frame.py --dataset-root C:\datasets\kitti_object --frame 000000
```
