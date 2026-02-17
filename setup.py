from setuptools import find_packages, setup

setup(
    name="mofa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "mediapipe",
        "trimesh",
        "pygltflib",
        "Pillow",
        "scipy",
    ],
)
