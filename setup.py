from setuptools import setup, find_packages

setup(
    name="visual-odometry",
    version="0.1.0",
    description="Monocular Visual Odometry with Pose Graph Optimization",
    author="Visual Odometry Project",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "opencv-contrib-python>=4.8",
        "open3d>=0.18",
        "matplotlib>=3.7",
        "graphslam>=0.4.0",
        "evo>=1.28",
        "sympy>=1.12",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        "rich>=13.0",
        "click>=8.1",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.1"],
        "notebook": ["jupyter>=1.0", "notebook>=7.0", "ipykernel>=6.0", "ipywidgets>=8.0"],
    },
    entry_points={
        "console_scripts": [
            "run-vo=scripts.run_vo:main",
            "evaluate-vo=scripts.evaluate:main",
        ],
    },
)
