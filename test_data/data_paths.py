# %%
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    _root_dir: Path = Path(__file__).parent.parent
    _test_data_dir: Path = _root_dir / "test_data"

    cat: Path = _test_data_dir / "cat.png"
    distorted_house: Path = _test_data_dir / "house.png"
    distorted_house_lens_model: Path = _test_data_dir / "house_lens_model.json"
    distorted_checkerboard: Path = _test_data_dir / "checkerboard.png"
    undistorted_checkerboard: Path = _test_data_dir / "undistorted_checkerboard.png"
    distorted_checkerboard_lens_model: Path = (
        _test_data_dir / "checkerboard_lens_model.json"
    )

    stereo_data_0_dir: Path = _test_data_dir / "stereo_data_0"
    stereo_data_1_dir: Path = _test_data_dir / "stereo_data_1"

    traproom1_dir: Path = _test_data_dir / "traproom1"

    ws_02_reference_data: Path = _test_data_dir / "ws_02_reference_data.npz"
    ws_03_reference_data: Path = _test_data_dir / "ws_03_reference_data.npz"
    ws_07_reference_data: Path = _test_data_dir / "ws_07_reference_data.npz"
