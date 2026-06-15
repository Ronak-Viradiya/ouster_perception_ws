# ouster_perception_ws

ROS workspace for Ouster lidar perception research and tools.

Contents
- `lidar_utils` — common utilities and helper functions used across packages.
- `pointcloud_publisher` — tools to publish example pointclouds / test data.
- `segmentation_inference` — segmentation model inference and helpers.

Quickstart

1. Install dependencies (system, ROS 2, Python packages) as appropriate for your platform.
2. From the workspace root, build with colcon:

```bash
colcon build --symlink-install
```

3. Source the local setup before running nodes or Python modules:

```bash
source install/setup.bash
```

Development notes
- The workspace uses `colcon` for builds. Packages live under `src/`.
- Editor settings in `.vscode/` were removed from history and are ignored by `.gitignore`.

Git / Collaboration
- If you previously cloned the repository before the recent history rewrite, re-clone to avoid divergent history:

```bash
git clone git@github.com:Ronak-Viradiya/ouster_perception_ws.git
```

Contributing
- Open an issue or send a pull request. Follow standard GitHub workflows; keep changes focused per package.

License
- See repository `LICENSE` if present; otherwise contact the maintainers for licensing information.

Maintainer
- Ronak Viradiya
