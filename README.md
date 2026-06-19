# ouster_perception_ws

Workspace for Ouster LiDAR perception research, tools and models.

**Overview**
- Provides utilities, data-publishing tools, and a segmentation inference package built as a ROS workspace (colcon).

**Top-level layout**
- `src/` — ROS / Python packages:
	- `lidar_utils/` — common utilities used across packages
	- `pointcloud_publisher/` — tools and launches to publish pointclouds and play recorded data
	- `segmentation_inference/` — model inference code and launches
- `scripts/` — helper scripts: `train.py`, `prepare_data.py`, `convert_cvat_to_labels.py`, etc.
- `data/` — datasets and exports (sequences, CVAT exports). Large files are ignored or tracked with Git LFS.
- `models/` — model scripts and `trained/` weights (kept out of normal git history or tracked with Git LFS).
- `rosbags/` — recorded ROS bag sessions for testing and playback

**Prerequisites**
- Ubuntu Jammy / ROS 2 (matching your target distro) and `colcon` installed.
- Python 3.10+ (recommended for ROS 2 Jammy). Use a venv for development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools
```



**Build (quick)**
1. From the workspace root:

```bash
colcon build --symlink-install
source install/setup.bash
```

2. To run package launch files (examples):

```bash
# play a recorded bag / publish pointclouds
ros2 launch pointcloud_publisher bag_playback.launch.py

# run segmentation inference
ros2 launch segmentation_inference segmentation.launch.py
```

If your ROS distro is different, adapt the `source` command accordingly.

**Data & models**
- Raw recorded sequences: `data/sequences/` and `rosbags/`.
- CVAT exports: `data/cvat_exports/` 
- Trained models: `models/trained/` 


**Scripts**
- `scripts/train.py` — training entrypoint (project-specific). Adjust dataset and hyperparameters inside the script or via CLI args.
- `scripts/prepare_data.py`, `scripts/convert_cvat_to_labels.py` — helpers to convert and prepare datasets from CVAT/other formats.

Run scripts from the workspace root with your activated venv, e.g.:

```bash
python3 scripts/prepare_data.py --input data/cvat_exports/sequence_00.zip --out data/preprocessed/
```

**Tests**
- Run package tests with colcon:

```bash
colcon test --packages-select segmentation_inference
colcon test-result --verbose
```

**Maintainer / Contact**
- Ronak Viradiya 

---
Last updated: 2026-06-19
