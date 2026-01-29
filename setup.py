from setuptools import setup, find_packages

setup(
    name="camie-tagger-quick",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "onnxruntime",
        "torchvision",
        "Pillow",
        "huggingface-hub",
    ],
    author="tamakara",
    description="A quick, out-of-the-box wrapper for CamieTagger-V2",
    python_requires=">=3.8",
)
