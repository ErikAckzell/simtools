#!/bin/sh

# Use correct virtual environment.
virtualenv=virtual_assimulo
py=${virtualenv}/bin/python

# Run code-extractor.
$py code_extractor.py

# Run BDF.py.
$py BDF.py

# Compile latex file.
cd TeX
pdflatex main.tex
