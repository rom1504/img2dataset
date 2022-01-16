from setuptools import setup, find_packages
from pathlib import Path

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name="img2dataset",
        packages=find_packages(),
        include_package_data=True,
        version="1.21.0",
        license="MIT",
        description="Easily turn a set of image urls to an image dataset",
        long_description=long_description,
        long_description_content_type="text/markdown",
        entry_points={"console_scripts": ["img2dataset = img2dataset:main"]},
        author="Romain Beaumont",
        author_email="romain.rom1@gmail.com",
        url="https://github.com/rom1504/img2dataset",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning", "computer vision", "download", "image", "dataset"],
        install_requires=[
            "tqdm",
            "opencv-python",
            "fire",
            "webdataset",
            "pandas",
            "pyarrow",
            "exifread",
            "albumentations",
            "pyyaml",
            "dataclasses",
            "wandb",
            "fsspec",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
        ],
    )
