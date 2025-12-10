from setuptools import setup, find_packages

setup(
    name="pravi-torch-view",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
    author="Pravi",
    description="A PyTorch tensor visualization tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
