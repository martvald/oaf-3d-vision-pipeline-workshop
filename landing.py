# %% [markdown]
# # Introduction
#
# Welcome to the "Pixel to Point Cloud" workshop repository! This project is designed
# to guide you through building a comprehensive 3D vision pipeline, transforming 2D
# images into 3D point clouds using Python and computer vision techniques.
#
# ## Overview
#
# This [repository](https://github.com/hariomshahu/oaf-3d-vision-pipeline-workshop/)
# contains materials for an 8-week workshop series that will introduce you to the
# fascinating world of 3D computer vision. Whether you're a beginner or have some
# programming experience, this course will help you understand and implement key
# concepts in 3D vision.
#
# ## Deliverables
#
# The following [workshop website](https://hariomshahu.github.io/oaf-3d-vision-pipeline-workshop/)
# is my deliverable for this project. Each participant will create their own fork of
# the repository and host their own version of the website.

# %% tags=["remove_input"]
import sys
from datetime import datetime

from oaf_vision_3d.tests.results import WorkshopResult

print(WorkshopResult())

# %% [markdown]
# ## Repository Structure
#
# - `workshops/`: Contains the jupyter notebooks for each workshop
# - `oaf_vision_3d/`: Python package we will build throughout the workshops
# - `test_data/`: Test images and data for the workshops
# - `docs/`: Files related to the documentation you are currently reading
#
# ## Workshop Schedule
#
# 1. [Introduction to 3D Vision](workshops/01_introduction_to_3d_vision.ipynb)
# 2. [Understanding Camera Models](workshops/02_understanding_camera_models.ipynb)
# 3. [Image Distortion and Undistortion](workshops/03_image_distortion_and_undistortion.ipynb)
# 4. [3D-2D Projections and PnP](workshops/04_3d_2d_projections_and_pnp.ipynb)
# 5. [Dual Camera Setups](workshops/05_dual_camera_setups.ipynb)
# 6. [Stereo Matching Fundamentals](workshops/06_stereo_matching_fundamentals.ipynb)
# 7. [Stereo Matching Fundamentals Continued](workshops/07_stereo_matching_fundamentals_continued.ipynb)
# 8. [Building Your 3D Vision Pipeline](workshops/08_building_your_3d_vision_pipeline.ipynb)
#
# ## Contributing
#
# We welcome contributions and suggestions! Please open an issue or submit a pull
# request if you have any improvements or find any bugs.
#
# ## License
#
# This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for
# details.
#
# Happy learning, and enjoy your journey into the world of 3D vision!
#
# ## Change history
#
# We don't keep history in this repository but the current version of these pages was
# made and published with:


# %% tags=["remove_input"]
print("- Last run:", datetime.now())
print("- Python version:", sys.version)
print("- OS:", sys.platform)
