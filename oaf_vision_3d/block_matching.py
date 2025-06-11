# %% [markdown]
# # Block Matching
#
# This function performs block matching to estimate the depth of a pixel in a set of 2D
# images. The process for this was discussed in more detail in the workshop
# [6: Stereo Matching Fundamentals](../workshops/06_stereo_matching_fundamentals.ipynb).


# %%
from enum import Enum

import numpy as np
from nptyping import Float32, Int32, NDArray, Shape
from scipy.signal import convolve2d

from oaf_vision_3d.poly_2_subvalue_fit import find_subvalue_poly_2


class CostFunction(Enum):
    SUM_OF_ABSOLUTE_DIFFERENCE = 0
    SUM_OF_SQUARED_DIFFERENCE = 1


def _get_cost(
    image_0: NDArray[Shape["H, W, ..."], Float32],
    image_1: NDArray[Shape["H, W, ..."], Float32],
    cost_function: CostFunction,
) -> NDArray[Shape["H, W"], Float32]:
    match cost_function:
        case CostFunction.SUM_OF_ABSOLUTE_DIFFERENCE:
            return np.abs(image_0 - image_1).sum(axis=-1)
        case CostFunction.SUM_OF_SQUARED_DIFFERENCE:
            return ((image_0 - image_1) ** 2).sum(axis=-1)
        case _:
            raise ValueError("Invalid cost function")


def block_matching(
    image_0: NDArray[Shape["H, W"], Float32],
    image_1: NDArray[Shape["H, W"], Float32],
    disparity_range: NDArray[Shape["2"], Float32],
    block_size: NDArray[Shape["[x, y]"], Int32] = np.array([11, 11], dtype=np.int32),
    subpixel_fit: bool = True,
    cost_function: CostFunction = CostFunction.SUM_OF_ABSOLUTE_DIFFERENCE,
) -> NDArray[Shape["H, W"], Float32]:
    disparities = np.arange(disparity_range[0], disparity_range[1], dtype=np.int32)
    error = []
    for _disparity in disparities:
        shifted_image_1 = np.roll(image_1, _disparity, axis=1)
        single_pixel_error = _get_cost(image_0, shifted_image_1, cost_function)

        convoluted_error = convolve2d(
            convolve2d(
                single_pixel_error,
                np.ones((1, block_size[0])) / block_size[0],
                mode="same",
            ),
            np.ones((block_size[1], 1)) / block_size[1],
            mode="same",
        )
        error.append(convoluted_error)

    disparity_error = np.array(error, dtype=np.float32)

    if subpixel_fit:
        disparity = find_subvalue_poly_2(
            values=disparities.astype(np.float32), function_value=disparity_error
        )
    else:
        disparity = disparities[np.argmin(disparity_error, axis=0)].astype(np.float32)

    disparity[:, : int(np.abs(disparities).max())] = np.nan
    disparity[:, -int(np.abs(disparities).max()) :] = np.nan
    disparity[disparity >= disparities.max()] = np.nan
    disparity[disparity <= disparities.min()] = np.nan

    return disparity
