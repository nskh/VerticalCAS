#!/usr/bin/env sh

brew install python@3.11

git submodule update --init --recursive
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r reqs.txt

cd GenerateNetworks/Marabou
pip install pybind11
brew install boost
brew install boost-python
brew install wget
