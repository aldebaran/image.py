
# Third-party libraries
import pytest
try:
	import cv2
	_has_CV = True
except ImportError:
	_has_CV = False

try:
	from Qt.QtGui import QPixmap, QImage
	_has_Qt = True
except ImportError:
	_has_Qt = False

# Local modules
from image import Image, Colorspace

def test_colorspace():
	rgb = Colorspace("RGB")
	assert("RGB" == rgb.name)
	assert(11 == rgb.al_code)
	assert(13 == rgb.qt_code)
	assert(3 == rgb.depth)
	assert(numpy.uint8 == rgb.dtype)

	assert(Colorspace("RGB") == Colorspace(Colorspace.RGB) == Colorspace.RGB)
	assert(Colorspace("RGB") == Colorspace(al_code=11) == Colorspace(qt_code=13))

def test_image_open_from_file(gray_file_path, color_file_path):
	i_color = Image(color_file_path)
	i_gray = Image(gray_file_path)
