Description
===========

Image compatibility wrapper around different tools (OpenCv, Qt, Naoqi)

This package provides a class allowing to create an image from different
frameworks and to convert it for other frameworks.

Currently supported frameworks are:
- OpenCV
- qi (Aldebaran's framework for robotics)
- Qt

Installation
============

For now, you can only install image.py from sources.

First, you need to install xmp (which can also only be installed from sources)::

	git clone https://github.com/aldebaran/xmp.git
	cd xmp
	python setup.py bdist_wheel
	pip install dist/*

Then install image.py::

	git clone https://github.com/aldebaran/image.py.git
	cd image.py
	python setup.py bdist_wheel
	pip install dist/*
