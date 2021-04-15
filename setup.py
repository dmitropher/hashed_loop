from setuptools import setup

setup(
    name="hashed_loop",
    version="0.1",
    description="Loop closure for use with Rosetta built with xbin and getpy",
    author="Dmitri Zorine",
    author_email="d.zorine@gmail.com",
    license="MIT",
    packages=["hashed_loop"],
    zip_safe=False,
    install_requires=["xbin", "getpy"],
)
