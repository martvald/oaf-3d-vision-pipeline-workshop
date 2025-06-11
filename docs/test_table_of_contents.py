from pathlib import Path


def _get_repo_path() -> Path:
    return Path(__file__).parent.parent.resolve()


def _get_all_markdown_files() -> list[Path]:
    markdown_files = []
    paths = _get_repo_path().rglob("*.py")
    for path in paths:
        if "dist" not in path.parts:
            lines = path.read_text(encoding="utf-8").splitlines()
            if "# %% [markdown]" in lines:
                markdown_files.append(path.parent / path.stem)

    paths = _get_repo_path().rglob("*.md")
    for path in paths:
        if "dist" not in path.parts:
            markdown_files.append(path.parent / path.stem)

    paths = _get_repo_path().rglob("*.ipynb")
    for path in paths:
        if "dist" not in path.parts:
            markdown_files.append(path.parent / path.stem)

    return markdown_files


def _get_all_toc_markdown_files() -> list[Path]:
    toc_path = _get_repo_path() / "docs/_toc.yml"
    lines = toc_path.read_text(encoding="utf-8").splitlines()

    files = []
    for line in lines:
        if "root: " in line:
            files.append(_get_repo_path() / line.strip()[6:])
        if "- file: " in line:
            files.append(_get_repo_path() / line.strip()[8:])

    return files


def _get_all_ignores_book_files() -> list[Path]:
    ignore_path = _get_repo_path() / "docs/.bookignore"
    with open(ignore_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    files = []
    for line in lines:
        globbed_files = _get_repo_path().rglob(line.strip())
        for globbed_file in globbed_files:
            files.append(globbed_file.parent / globbed_file.stem)

    return files


def test_toc() -> None:
    markdown_files = _get_all_markdown_files()
    toc_files = _get_all_toc_markdown_files()
    ignore_files = _get_all_ignores_book_files()

    # Check that all calibration markdown files are included in toc or ignored
    for markdown_file in markdown_files:
        if (markdown_file not in toc_files) and (markdown_file not in ignore_files):
            raise AssertionError(
                f"{markdown_file.relative_to(_get_repo_path()).as_posix()} not found in _toc.yml or .bookignore"
            )

    # Check that all files in toc actually exist
    for toc_file in toc_files:
        if toc_file not in markdown_files:
            raise AssertionError(
                f"{toc_file.relative_to(_get_repo_path()).as_posix()} does not exist, but is referenced in _toc.yml"
            )

    # Check that ignored file are not included in toc
    for ignore_file in ignore_files:
        if ignore_file in toc_files:
            raise AssertionError(
                f"{ignore_file.relative_to(_get_repo_path()).as_posix()} exist in .bookignore, but is referenced in _toc.yml"
            )


if __name__ == "__main__":
    test_toc()
