
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
from image import Image

def test_image_open(jpg_file_path):
	i = Image(jpg_file_path)
	if _has_CV: assert("CVImage" == i.type)
	elif _has_Qt: assert("QImage" == i.type)