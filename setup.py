import setuptools
import glob
neigh_indices_files = glob.glob('sphericalunet/utils/neigh_indices/*')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sphericalunet",
    version="v1.0.5",
    author="Fenqiang Zhao",
    author_email="zhaofenqiang0221@gmail.com",
    description="This is the tools for Spherical U-Net on spherical cortical surfaces",
    long_description="long_description",
    long_description_content_type="text/markdown",
    url="https://github.com/zhaofenqiang/Spherical_U-Net",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    data_files=[('neigh_indices', neigh_indices_files)],
)

