#!/bin/bash -e
# Publish contents of dist folder to PyPI.
# $1: PyPI password
# $2: PyPI repository to use (pypi or testpypi)

python -m pip install --upgrade twine==3.2.0
python -m twine check dist/*
python -m twine upload -u __token__ -p "$1" -r "$2" dist/*
echo Uploaded to PyPI.
