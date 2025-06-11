# %% [markdown]
# # Polyfit 2 subvalue local minima

# %%
import numpy as np
from nptyping import Float32, NDArray, Shape


def find_subvalue_poly_2(
    values: NDArray[Shape["N"], Float32],
    function_value: NDArray[Shape["N, H, W"], Float32],
) -> NDArray[Shape["H, W"], Float32]:
    h_idx = np.arange(function_value.shape[1])
    w_idx = np.arange(function_value.shape[2])
    h_idx, w_idx = np.meshgrid(h_idx, w_idx, indexing="ij")

    idx = np.clip(np.argmin(function_value, axis=0), 1, values.shape[0] - 2)

    f_0 = function_value[idx - 1, h_idx, w_idx]
    f_1 = function_value[idx, h_idx, w_idx]
    f_2 = function_value[idx + 1, h_idx, w_idx]

    a = 0.5 * (f_0 + f_2) - f_1
    b = 0.5 * (f_2 - f_0)

    denom = 2 * a
    denom = np.where(denom == 0, np.nan, denom)

    delta = -b / denom
    delta = np.where(np.abs(delta) > 1, np.nan, delta)

    return values[idx] + delta
