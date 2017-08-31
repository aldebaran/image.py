
# Third-party libraries
import numpy
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

def test_image_open_from_file(gray_file_path, color_file_path, depth_file_path):

	if _has_CV:
		i_color = Image(color_file_path)
		i_gray = Image(gray_file_path)
		i_depth = Image(depth_file_path)
		# i_gray is Gray when opened with OpenCV
		assert(Colorspace("Gray") == i_gray.colorspace)
		# i_color is BGR if opened with OpenCV
		assert(Colorspace("BGR") == i_color.colorspace)
		# i_depth is Depth if opened with OpenCV
		assert(Colorspace("Depth") == i_depth.colorspace)
		assert((640,480) == i_gray.resolution)

		# Test depth rendering
		i_depth_rendered = i_depth.render()
		assert(Colorspace("BGR") == i_depth_rendered.colorspace)

		# Test colorspace assignement and saving
		depth_file_copy_path = depth_file_path+"_copy.png"
		i_depth.colorspace = "Distance"
		i_depth.save(depth_file_copy_path)
		i_depth2 = Image(depth_file_copy_path)
		assert(Colorspace("Distance") == i_depth2.colorspace)

		# Test cv_image attribute
		assert(isinstance(i_color.cv_image, numpy.ndarray))
		assert(isinstance(i_gray.cv_image, numpy.ndarray))
		assert(isinstance(i_depth.cv_image, numpy.ndarray))

		# Test qimage attribute
		if not _has_Qt:
			assert(not hasattr(i_color, "qimage"))
		else:
			assert(hasattr(i_color, "qimage"))
			assert(isinstance(i_color.qimage, QImage))
			assert(isinstance(i_gray.qimage, QImage))
			assert(isinstance(i_depth.qimage, QImage))

	elif _has_Qt:
		i_color = Image(color_file_path)
		i_gray = Image(gray_file_path)
		i_depth = Image(depth_file_path)
		assert(not hasattr(i_color, "cv_image"))

		# i_color is RGB if opened with Qt
		assert(Colorspace("RGB") == i_color.colorspace)
		# i_gray is Gray if opened with Qt
		assert(Colorspace("Gray") == i_gray.colorspace)
		# i_depth is Gray if opened with OpenCV
		assert(Colorspace("Gray") == i_depth.colorspace)

		# Test depth rendering is not possible
		with pytest.raises(RuntimeError):
			i_depth_rendered = i_depth.render()

	else:
		with pytest.raises(RuntimeError):
			i_depth = Image(depth_file_path)

@pytest.mark.skipif(not _has_CV and not _has_Qt,
                    reason="Image cannot be opened without Qt nor cv2")
def test_image_camera_matrix_set(left_file_path):
	i_left = Image(left_file_path)
	assert(1.0 == i_left.camera_info.camera_matrix[0][0])

	i_left.camera_info._camera_matrix[0][0] = 10.0
	assert(10.0 == i_left.camera_info.camera_matrix[0][0])
	i_left.save(left_file_path)

	i_left2 = Image(left_file_path)
	assert(10.0 == i_left2.camera_info.camera_matrix[0][0])
