import os
from pathlib import Path

from jupyter_book.cli.main import build

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    os.environ["JUPYTER_BOOK_BUILD"] = "1"
    build.main(
        [
            str(repo_path),
            "--config",
            str(repo_path / "docs" / "_config.yml"),
            "--toc",
            str(repo_path / "docs" / "_toc.yml"),
            "--warningiserror",
        ]
    )
