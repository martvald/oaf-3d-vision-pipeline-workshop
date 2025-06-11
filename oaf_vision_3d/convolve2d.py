# %% [markdown]
# # Convolution 2D with NaN
#
# This function convolves a 2D image with a given kernel. If the kernel is 1D, it is
# convolved first along the vertical axis and then along the horizontal axis. If the
# kernel is 2D, it is convolved directly. The function also handles NaN values in the
# image by setting them to 0 before convolution and then dividing the result by the
# number of non-NaN values in the kernel.

# %%
import warnings

import numpy as np
from nptyping import Float32, NDArray, Shape
from scipy.signal import convolve2d


def convolution_2d_nan(
    image: NDArray[Shape["H, W"], Float32], kernel: NDArray[Shape["K, ..."], Float32]
) -> NDArray[Shape["H, W"], Float32]:
    valid_image = image.copy()

    isnans = np.isnan(valid_image)
    valid_image[isnans] = 0

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
    )
    match kernel.ndim:
        case 1:
            convolved_valid_image = convolve2d(
                convolve2d(
                    valid_image,
                    kernel[:, None],
                    mode="same",
                ),
                kernel[None, :],
                mode="same",
            )
            convolved_no_nan = convolve2d(
                convolve2d(
                    np.logical_not(isnans).astype(np.float32),
                    kernel[:, None],
                    mode="same",
                ),
                kernel[None, :],
                mode="same",
            )
            return convolved_valid_image / convolved_no_nan
        case 2:
            convolved_valid_image = convolve2d(
                valid_image,
                kernel,
                mode="same",
            )
            convolved_no_nan = convolve2d(
                np.logical_not(isnans).astype(np.float32),
                kernel,
                mode="same",
            )
            return convolved_valid_image / convolved_no_nan

        case _:
            raise ValueError("Kernel must be 1D or 2D")
