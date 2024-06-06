from setuptools import find_packages, setup

setup(
    name="LLRS",
    version="1.0.0",
    url="https://github.com/TQT-RAAQS/LLRS",
    packages=find_packages(),
   # install_requires=["numpy", "sympy", "scipy", "matplotlib"],
    python_requires=">=3",

    # script generation
    entry_points={}
)
