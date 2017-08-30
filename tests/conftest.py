#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
#                            SOFTBANK  ROBOTICS
#==============================================================================
# PROJECT : image wrapper
# FILE : conftest.py
# DESCRIPTION :
"""
Prepare the conditions for proper unit testing
"""
#[MODULES IMPORTS]-------------------------------------------------------------
import os
import errno
import shutil
import pytest

#[MODULE INFO]-----------------------------------------------------------------
__author__ = "sambrose"
__date__ = "2017-08-23"
__copyright__ = "Copyright 2017, Softbank Robotics (c)"
__version__ = "1.0.0"
__maintainer__ = "sambrose"
__email__ = "sambrose@softbankrobotics.com"

#[MODULE GLOBALS]--------------------------------------------------------------

DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
SANDBOX_FOLDER = "/tmp/image_wrapper/"

GRAY_PHOTO = "top.png"
COLOR_PHOTO = "SpringNebula.jpg"

#[MODULE CONTENT]--------------------------------------------------------------

def sandboxed(path):
	"""
	Makes a copy of the given path in /tmp and returns its path.
	"""
	source_path = os.path.join(DATA_FOLDER,    path)
	tmp_path    = os.path.join(SANDBOX_FOLDER, path)

	try:
		os.mkdir(SANDBOX_FOLDER)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	if os.path.isdir(source_path):
		if os.path.exists(tmp_path):
			shutil.rmtree(tmp_path)
		shutil.copytree(source_path, tmp_path)
	else:
		shutil.copyfile(source_path, tmp_path)

	return tmp_path

@pytest.fixture(autouse=True, scope="function")
def begin(request):
	"""
	Add a finalizer to clean tmp folder after each test
	"""
	def fin():
		if os.path.exists(SANDBOX_FOLDER):
			shutil.rmtree(SANDBOX_FOLDER)

	request.addfinalizer(fin)

@pytest.fixture(scope="function")
def color_file_path():
	return sandboxed(COLOR_PHOTO)

@pytest.fixture(scope="function")
def gray_file_path():
	return sandboxed(GRAY_PHOTO)

