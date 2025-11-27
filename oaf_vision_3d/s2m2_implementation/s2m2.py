# %% [markdown]
# # S2M2
#
# This is a wrapper for the [S2M2](https://github.com/junhong-3dv/s2m2/tree/main) model.
#
# It is a PyTorch implementation of the model that can be used to estimate depth from stereo images.
#
# ## Documentation
#
# Please refere to the [README](./README.md) and [LICENSE](./LICENSE.md) for more details.
#
# ## Pretrained Models
#
# Pretrained models are availible in the [README](./README.md):
#
# | Model  |                                  Download                                  | Model Size |
# |:------:|:--------------------------------------------------------------------------:|:----------:|
# | **S**  | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH128NTR1.pth) |   26.5M    |
# | **M**  | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH192NTR2.pth) |   80.4M    |
# | **L**  | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH256NTR3.pth) |    181M    |
# | **XL** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth) |    406M    |
#
# To use with `oaf_vision_3d`, you need to download the desired model(s) and place them
# in repository root directory under `./pretrained_models/S2M2`. Rename the files to
# `S.pth`, `M.pth`, `L.pth`, and `XL.pth`, f.ex. `./pretrained_models/S2M2/S.pth`. See
# [pretrained_paths.py](./pretrained_paths.py) for the exact paths.
#
# Then, you can use the model like this:
#
# ```python
# model = load_model(model_config=S2M2Config.small())
# ```

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nptyping import Float32, NDArray, Shape, UInt8
from torch.amp.autocast_mode import autocast
from torch.utils import dlpack as torch_dlpack

from oaf_vision_3d.s2m2_implementation.model.s2m2 import S2M2
from oaf_vision_3d.s2m2_implementation.pretrained_paths import PretrainedPaths

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class S2M2Config:
    checkpoint_path: Path
    feature_channels: int = 384
    dim_expansion: int = 1
    num_transformer: int = 3
    use_positivity: bool = True
    refine_iter: int = 3

    @staticmethod
    def small() -> S2M2Config:
        return S2M2Config(
            checkpoint_path=PretrainedPaths.s2m2_model_s,
            feature_channels=128,
            num_transformer=1,
        )

    @staticmethod
    def medium() -> S2M2Config:
        return S2M2Config(
            checkpoint_path=PretrainedPaths.s2m2_model_m,
            feature_channels=192,
            num_transformer=2,
        )

    @staticmethod
    def large() -> S2M2Config:
        return S2M2Config(
            checkpoint_path=PretrainedPaths.s2m2_model_l,
            feature_channels=256,
            num_transformer=3,
        )

    @staticmethod
    def extra_large() -> S2M2Config:
        return S2M2Config(
            checkpoint_path=PretrainedPaths.s2m2_model_xl,
            feature_channels=384,
            num_transformer=3,
        )


def load_model(
    model_config: S2M2Config = S2M2Config.small(),
    allow_negative: bool = True,
    torch_compile: bool = False,
) -> S2M2:
    if not model_config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {model_config.checkpoint_path}"
        )

    model = S2M2(
        feature_channels=model_config.feature_channels,
        dim_expansion=1,
        num_transformer=model_config.num_transformer,
        use_positivity=not allow_negative,
        refine_iter=model_config.refine_iter,
    )
    checkpoint = torch.load(
        model_config.checkpoint_path,
        weights_only=True,
        map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    model.my_load_state_dict(checkpoint["state_dict"])

    if torch_compile:
        model: S2M2 = torch.compile(model)  # type: ignore[assignment, no-redef]

    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return model


def _get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clean_disparity(
    disparity: NDArray[Shape["H, W"], Float32],
) -> NDArray[Shape["H, W"], Float32]:
    offset = np.arange(disparity.shape[1])[None] - disparity
    valid = np.logical_and(offset >= 0, offset <= disparity.shape[1] - 1)
    return np.where(valid, disparity, np.nan)


def infer(
    model: S2M2,
    left: NDArray[Shape["H, W, 3"], UInt8],
    right: NDArray[Shape["H, W, 3"], UInt8],
    device: torch.device = _get_best_device(),
) -> tuple[
    NDArray[Shape["H, W"], Float32],
    NDArray[Shape["H, W"], Float32],
    NDArray[Shape["H, W"], Float32],
    NDArray[Shape["H, W"], Float32],
]:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    raw_img_height, raw_img_width = left.shape[:2]
    img_height = (raw_img_height // 32) * 32
    img_width = (raw_img_width // 32) * 32

    # image crop
    left = left[:img_height, :img_width]
    right = right[:img_height, :img_width]
    left_torch = torch_dlpack.from_dlpack(left.transpose(2, 0, 1)[None])
    right_torch = torch_dlpack.from_dlpack(right.transpose(2, 0, 1)[None])

    with torch.no_grad():
        with autocast(enabled=True, device_type=device.type, dtype=torch.float):
            disparity, occlusion, confidence = model(left_torch, right_torch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            disparity = F.pad(
                disparity,
                (
                    0,
                    raw_img_width - img_width,
                    0,
                    raw_img_height - img_height,
                ),
                value=torch.nan,
            )
            occlusion = F.pad(
                occlusion,
                (
                    0,
                    raw_img_width - img_width,
                    0,
                    raw_img_height - img_height,
                ),
                value=torch.nan,
            )
            confidence = F.pad(
                confidence,
                (
                    0,
                    raw_img_width - img_width,
                    0,
                    raw_img_height - img_height,
                ),
                value=torch.nan,
            )
            mask = (confidence.cpu().float() > 0.1).squeeze()

    return (
        _clean_disparity(disparity[0, 0].cpu().numpy()),
        occlusion[0, 0].cpu().numpy(),
        confidence[0, 0].cpu().numpy(),
        mask[0, 0].cpu().numpy(),
    )
