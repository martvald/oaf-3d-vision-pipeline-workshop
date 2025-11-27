# %% [markdown]
# # Pretrained Paths
#
# The `PretrainedPaths` dataclass provides a centralized and immutable configuration for
# accessing various paths used to load pretrained models and other resources.


# %%
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PretrainedPaths:
    _repo_path = (
        Path(__file__).parent.parent.parent
        if "__file__" in globals().keys()
        else Path("vision3d").resolve()
    )
    _pretrained_models_dir: Path = _repo_path / "pretrained_models"

    _s2m2_dir: Path = _pretrained_models_dir / "S2M2"
    s2m2_model_s: Path = _s2m2_dir / "S.pth"
    s2m2_model_m: Path = _s2m2_dir / "M.pth"
    s2m2_model_l: Path = _s2m2_dir / "L.pth"
    s2m2_model_xl: Path = _s2m2_dir / "XL.pth"
