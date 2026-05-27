"""
HydraLab-USV package setup.

Install in development mode:
    pip install -e .

For full training support:
    pip install -e ".[train]"

For Isaac Lab (requires Isaac Sim installation first):
    pip install -e ".[isaac]"
"""

from setuptools import setup, find_packages

setup(
    name="hydralab-usv",
    version="0.1.0",
    description="GPU-accelerated RL environment for autonomous surface vessels",
    author="HydraLab",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "gymnasium>=0.29",
    ],
    extras_require={
        "train": [
            "stable-baselines3>=2.0",
            "tensorboard>=2.13",
        ],
        "viz": [
            "matplotlib>=3.7",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "isaac": [
            # Isaac Sim / Isaac Lab are installed separately via omniverse
            # See: https://isaac-sim.github.io/IsaacLab/
        ],
        "all": [
            "stable-baselines3>=2.0",
            "tensorboard>=2.13",
            "matplotlib>=3.7",
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
