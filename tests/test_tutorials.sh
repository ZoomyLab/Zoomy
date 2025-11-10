# !/bin/bash

jupytext --to py tutorials/swe/*.ipynb
pytest --html=pytest-report.html --self-contained-html tutorials/swe/*.py -m nbworking
mv pytest-report.html web/


