python_sources(name="root")

python_distribution(
    name="dist",
    dependencies=["//oaf_vision_3d"],
    provides=python_artifact(name="oaf-vision-3d", version="0.1.0"),
    description="Python package for OAF Vision 3D.",
)

files(name="markdowns", sources=["**/*.md"])
files(name="workshops", sources=["workshops/*.ipynb"])
