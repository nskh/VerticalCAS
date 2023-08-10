from setuptools import setup, find_packages

setup(
    name='VerticalCAS',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # I think we can leave this empty by building the conda environment first
    ],
)
