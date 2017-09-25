
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
from image import Image, Colorspace, CameraInfo

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
	assert([] == i_left.camera_info.distortion_coeffs)

	i_left.camera_info._camera_matrix[0][0] = 10.0
	assert(10.0 == i_left.camera_info.camera_matrix[0][0])
	i_left.camera_info._distortion_coeffs.append(-0.1)
	assert(-0.1 == i_left.camera_info.distortion_coeffs[0])
	i_left.save(left_file_path)

	i_left2 = Image(left_file_path)
	assert(10.0 == i_left2.camera_info.camera_matrix[0][0])
	assert(-0.1 == i_left.camera_info.distortion_coeffs[0])

@pytest.mark.skipif(not _has_CV and not _has_Qt,
                    reason="Image cannot be opened without Qt nor cv2")
def test_image_stereo(stereo_file_path):
	# Open image
	i_stereo = Image(stereo_file_path)
	assert(not hasattr(i_stereo, "left_image"))
	assert(not hasattr(i_stereo, "right_image"))
	assert(not hasattr(i_stereo, "top_image"))
	assert(not hasattr(i_stereo, "bottom_image"))
	assert(i_stereo.is_mono)

	# Create left and right camera info
	mono_cam_info = CameraInfo()
	mono_cam_info.setWidth(2560)
	mono_cam_info.setHeight(720)
	left_cam_info = CameraInfo()
	left_cam_info.setWidth(1280)
	left_cam_info.setHeight(720)
	right_cam_info = CameraInfo()
	right_cam_info.setWidth(1280)
	right_cam_info.setHeight(720)
	top_cam_info = CameraInfo()
	top_cam_info.setWidth(2560)
	top_cam_info.setHeight(360)
	bottom_cam_info = CameraInfo()
	bottom_cam_info.setWidth(2560)
	bottom_cam_info.setHeight(360)

	# Set image as vertical stereo
	i_stereo.setIsStereo(top_cam_info, bottom_cam_info)
	assert(hasattr(i_stereo, "top_image"))
	assert(hasattr(i_stereo, "bottom_image"))
	assert((2560,360) == i_stereo.top_image.resolution)
	assert((2560,360) == i_stereo.bottom_image.resolution)
	assert(not hasattr(i_stereo, "left_image"))
	assert(not hasattr(i_stereo, "right_image"))

	# Change to horizontal stereo
	i_stereo.setIsStereo(left_cam_info, right_cam_info)
	assert(hasattr(i_stereo, "left_image"))
	assert(hasattr(i_stereo, "right_image"))
	assert((1280,720) == i_stereo.left_image.resolution)
	assert((1280,720) == i_stereo.right_image.resolution)
	assert(not hasattr(i_stereo, "top_image"))
	assert(not hasattr(i_stereo, "bottom_image"))

	# Save and check if changes were saved
	i_stereo.save(stereo_file_path)

	i_stereo = Image(stereo_file_path)
	assert(1280 == i_stereo.camera_info[0].width)
	assert(720 == i_stereo.camera_info[0].height)
	assert(1280 == i_stereo.camera_info[1].width)
	assert(720 == i_stereo.camera_info[1].height)
	assert(2560 == i_stereo.width)
	assert(720 == i_stereo.height)
	assert(hasattr(i_stereo, "left_image"))
	assert(hasattr(i_stereo, "right_image"))
	assert(i_stereo.is_stereo)

	# Go back to mono image
	i_stereo.setIsMono(mono_cam_info)
	assert(not hasattr(i_stereo, "left_image"))
	assert(not hasattr(i_stereo, "right_image"))
	assert(not hasattr(i_stereo, "top_image"))
	assert(not hasattr(i_stereo, "bottom_image"))

	# Give bad camera infos
	bad_left_cam_info = CameraInfo()
	bad_right_cam_info = CameraInfo()
	bad_top_cam_info = CameraInfo()
	bad_bottom_cam_info = CameraInfo()

	with pytest.raises(RuntimeError) as _e:
		i_stereo.setIsStereo(bad_top_cam_info, bad_bottom_cam_info)
	assert("Images given sizes ((0,0) and (0,0)) don't match original image size (2560,720)"\
	        == _e.value.message)
	assert(i_stereo.is_mono)

	with pytest.raises(RuntimeError) as _e:
		i_stereo.setIsStereo(bad_left_cam_info, bad_right_cam_info)
	assert("Images given sizes ((0,0) and (0,0)) don't match original image size (2560,720)"\
	        == _e.value.message)
	assert(i_stereo.is_mono)
