from setuptools import setup, find_packages

setup(
    name="legged_gym",
    version="1.0.0",
    description="Isaac Gym Humanoid Robot Complex Terrain Locomotion",
    author="Humanoid Locomotion Team",
    author_email="example@email.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pyyaml",
        "tensorboard",
        "opencv-python",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 