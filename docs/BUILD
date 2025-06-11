python_sources()

python_source(
    name="build_docs",
    source="build_docs.py",
    run_goal_use_sandbox=False,
    dependencies=[
        "//:root",
        "//oaf_vision_3d",
        "//third_party/python:reqs",
        "//:markdowns",
        "//test_data:test_data_files",
        "//:workshops",
    ],
    description="Builds the documentation and website using Jupyter Book.",
)

python_tests(
    name="tests",
    dependencies=[":build_docs", ":_toc", ":bookignore"],
)

file(
    name="_toc",
    source="_toc.yml",
)
file(
    name="bookignore",
    source=".bookignore",
)
