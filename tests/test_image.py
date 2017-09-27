# Copyright (c) 2017, Softbank Robotics Europe
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
	i_left.camera_info.setDistortionCoeffs([-0.1])
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

def test_load_camera_info_from_naoqi_file(calibration_file_path):
	info_from_cal_file = CameraInfo.fromNaoqiCalibrationFile(calibration_file_path)

	def isclose(a, b, rel_tol=1e-08, abs_tol=0.0):
		return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

	def is_equal(a,b):
		if len(a) != len(b):
			return False
		for i in range(len(a)):
			if isinstance(a[i], list):
				if not is_equal(a[i],b[i]):
					return False
			else:
				if not isclose(a[i], b[i]):
					print a[i], b[i], abs(a[i]-b[i]), 1e-09 * max(abs(a[i]), abs(b[i]))
					return False
		return True

	assert(1280 == info_from_cal_file.width)
	assert(720 == info_from_cal_file.height)
	assert(
	    is_equal(
	        [
	            [6.96196228e+02, 0., 6.50011292e+02],
	            [0., 6.99801025e+02, 3.68380310e+02],
	            [0., 0., 1.]
	        ],
	        info_from_cal_file.camera_matrix
	    )
	)
	assert(
	    is_equal(
	        [
	            -1.63907841e-01,
	            9.65181086e-03,
	            -3.43979977e-04,
	            -8.26001575e-04,
	            1.07839936e-02
	        ],
	        info_from_cal_file.distortion_coeffs
	    )
	)
	assert(
	    is_equal(
	        [
	            [9.84224379e-01, 5.37100509e-02, -1.68575093e-01],
	            [-5.43872640e-02, 9.98519719e-01, 6.00773201e-04],
	            [1.68357834e-01, 8.57704226e-03, 9.85688627e-01]
	        ],
	        info_from_cal_file.rectification_matrix
	    )
	)
	assert(
	    is_equal(
	        [
	            [5.73509216e+02, 0., 8.86482422e+02, 0.],
	            [0., 5.73509216e+02, 3.82152710e+02, 0.],
	            [0., 0., 1., 0.]
	        ],
	        info_from_cal_file.projection_matrix
	    )
	)
