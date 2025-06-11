import os
from typing import Optional

from matplotlib import pyplot as plt
from nptyping import Float32, NDArray, Shape


def is_in_jupyter_build() -> bool:
    print(os.environ.get("JUPYTER_BOOK_BUILD", "Does not exist"))
    if "JUPYTER_BOOK_BUILD" in os.environ:
        return True
    return False


def simple_plot(
    data: NDArray[Shape["H, W"], Float32],
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 6),
    colorbar: bool = True,
    axis_off: bool = True,
    thight_layout: bool = True,
) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(data)
    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)
    if axis_off:
        plt.axis("off")
    if thight_layout:
        plt.tight_layout()
    plt.show()
