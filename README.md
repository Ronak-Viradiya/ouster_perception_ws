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

- Git LFS must be installed and enabled (this repository uses Git LFS for large assets):

```bash
sudo apt update && sudo apt install git-lfs -y
git lfs install
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
- CVAT exports: `data/cvat_exports/` (note: `.zip` exports are large and ignored by default — check `.gitignore`).
- Trained models: `models/trained/` — large checkpoints are tracked with Git LFS or stored externally.

If you need to add a large data file, either copy it outside the repo or track it with Git LFS:

```bash
git lfs track "path/to/large-file"
git add .gitattributes
git add path/to/large-file
git commit -m "Add large asset to Git LFS"
git push
```

**Scripts**
- `scripts/train.py` — training entrypoint (project-specific). Adjust dataset and hyperparameters inside the script or via CLI args.
- `scripts/prepare_data.py`, `scripts/convert_cvat_to_labels.py` — helpers to convert and prepare datasets from CVAT/other formats.

Run scripts from the workspace root with your activated venv, e.g.:

```bash
python3 scripts/prepare_data.py --input data/cvat_exports/sequence_00.zip --out data/preprocessed/
```

**Git, history & collaboration**
- This repo had a large-file history rewrite to move big assets into Git LFS. If you cloned before that change, you must reclone or reset:

```bash
# easiest: fresh clone
git clone git@github.com:Ronak-Viradiya/ouster_perception_ws.git

# or, in an existing clone (will discard local commits):
git fetch origin
git reset --hard origin/master
git lfs install
```

- Make sure collaborators run `git lfs install` after cloning so LFS objects are pulled correctly.

**Troubleshooting**
- Push rejected due to large file: ensure the file is tracked with LFS or removed from history before pushing.
- If `colcon build` fails, check that ROS environment is sourced and required system packages are installed.

**Tests**
- Run package tests with colcon:

```bash
colcon test --packages-select segmentation_inference
colcon test-result --verbose
```

**Maintainer / Contact**
- Ronak Viradiya — open an issue or PR for changes.

---
Last updated: 2026-06-19
