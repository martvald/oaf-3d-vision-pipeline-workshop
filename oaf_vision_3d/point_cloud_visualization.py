# %% [markdown]
# # Point Cloud Visualization
#
# This module provides functions to visualize 3D point clouds using `open3d`.

# %% tags=["raises-exception", "remove-output"]
import numpy as np
from nptyping import Float32, NDArray, Shape

from oaf_vision_3d._notebook_tools import is_in_jupyter_build


def open3d_visualize_point_cloud(
    xyz: NDArray[Shape["H, W, 3"], Float32], rgb: NDArray[Shape["H, W, 3"], Float32]
) -> None:
    if not is_in_jupyter_build():
        # pylint: disable=import-outside-toplevel
        import open3d as o3d  # type: ignore

        points = xyz.reshape(-1, 3)
        points[:, 1:3] *= -1
        colors = rgb.reshape(-1, 3)
        invalid = np.isnan(points).any(axis=1)
        points = points[~invalid]
        colors = colors[~invalid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()  # type: ignore[reportAttributeAccessIssue]
        vis.create_window()
        vis.add_geometry(pcd)

        render_options = vis.get_render_option()
        render_options.point_size = 2

        vis.run()
        vis.destroy_window()
