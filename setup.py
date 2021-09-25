# import setuptools
from setuptools import setup

# import distutils.command.build
# from setuptools.command.install import install

# copy store imports
# import os
# from shutil import copyfile


setup(
    name="hashed_loop",
    version="0.1",
    description="Loop closure for use with Rosetta built with xbin and getpy",
    author="Dmitri Zorine",
    author_email="d.zorine@gmail.com",
    license="MIT",
    packages=["hashed_loop"],
    zip_safe=False,
    install_requires=["click", "xbin", "getpy", "h5py", "npose"],
    entry_points={
        "console_scripts": [
            "import_default_loop_table = hashed_loop.import_default_loop_table:main",
            "close_loop = hashed_loop.close_loop.close_loop:main",
            "build_hash_loop_table = hashed_loop.build_hash_loop_table:main",
        ]
    },
)
