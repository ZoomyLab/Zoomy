# !/bin/bash

jupytext --to py tutorials/swe/*.ipynb
pytest tutorials/swe/*.py -m nbworking

