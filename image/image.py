# -*- coding: utf-8 -*-

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
#                          © Softbank Robotics Europe                          #
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#

"""
Constants related to Naoqi images and conversion tools with CVImage, QImage.

The Image class is dedicated to the conversion of images between ALImage,
CVImage and QImage formats as well as image conversion. It thus depends on the
presence of PySide and/or cv2 to perform its functions and is only defined if
those packages are available.
"""

import fx.optional as optional
# Standard Library
import collections
import os
# Third-parties
import numpy as numpy
if optional.PySide: from PySide.QtGui import QPixmap, QImage
if optional.cv2: import cv2

# ──────────────────────── Embedded vision definitions ─────────────────────── #

class Colorspace:
	# –––––––––
	# Constants

	NAME_FROM_CODE = {
		 0: "Yuv",
		 1: "yUv",
		 2: "yuV",
		 3: "Rgb",
		 4: "rGb",
		 5: "rgB",
		 6: "Hsy",
		 7: "hSy",
		 8: "hsY",
		 9: "YUV422",
		10: "YUV",
		11: "RGB",
		12: "HSY",
		13: "BGR",
		14: "YYCbCr",
		15: "H2RGB",
		16: "HSMixed",
		17: "Depth",
		18: "ARGB",
		19: "XYZ",
		20: "Infrared",
		21: "Distance",
		22: "Lab",
		23: "RawDepth",
		24: "Luv",
		25: "LChab",
		26: "LChuv"
	}

	COLOR    = set(range(17)+[18,22,24,25,26])
	DEPTH    = set([17,19,21,23])
	INFRARED = set([20])

	DEPTH_FROM_COLORSPACE_NAME = {
		"Yuv"      : 1,
		"yUv"      : 1,
		"yuV"      : 1,
		"Rgb"      : 1,
		"rGb"      : 1,
		"rgB"      : 1,
		"Hsy"      : 1,
		"hSy"      : 1,
		"hsY"      : 1,
		"YUV422"   : 2,
		"YUV"      : 3,
		"RGB"      : 3,
		"HSY"      : 3,
		"BGR"      : 3,
		"YYCbCr"   : 2,
		"H2RGB"    : 3,
		"HSMixed"  : 3,
		"Depth"    : 1,
		"ARGB"     : 4,
		"XYZ"      : 3,
		"Infrared" : 1,
		"Distance" : 1,
		"Lab"      : 3,
		"RawDepth" : 1,
		"Luv"      : 3,
		"LChab"    : 3,
		"LChuv"    : 3,
		# The following colorspace is not provided by ALImage
		"Gray" : 1
	}

	_SAMPLE_TYPE_FROM_COLORSPACE_NAME = {
		"Yuv"      : numpy.uint8,
		"yUv"      : numpy.uint8,
		"yuV"      : numpy.uint8,
		"Rgb"      : numpy.uint8,
		"rGb"      : numpy.uint8,
		"rgB"      : numpy.uint8,
		"Hsy"      : numpy.uint8,
		"hSy"      : numpy.uint8,
		"hsY"      : numpy.uint8,
		"YUV422"   : numpy.uint8,
		"YUV"      : numpy.uint8,
		"RGB"      : numpy.uint8,
		"HSY"      : numpy.uint8,
		"BGR"      : numpy.uint8,
		"YYCbCr"   : numpy.uint8,
		"H2RGB"    : numpy.uint8,
		"HSMixed"  : numpy.uint8,
		"Depth"    : numpy.uint16,
		"ARGB"     : numpy.uint8,
		"XYZ"      : numpy.float32,
		"Infrared" : numpy.uint16,
		"Distance" : numpy.uint16,
		"Lab"      : numpy.uint8,
		"RawDepth" : numpy.uint16,
		"Luv"      : numpy.uint8,
		"LChab"    : numpy.uint8,
		"LChuv"    : numpy.uint8,
		# The following colorspace is not provided by ALImage
		"Gray"     : numpy.uint8
	}

	# ––––––––––––––
	# Static Helpers

	@staticmethod
	def allKnownNames():
		return list(set(Colorspace.NAME_FROM_CODE.values()))

	# –––––––––––
	# Constructor

	def __init__(self, spec):
		"""
		Constructor from colorspace code or name.

		Codes should be passed as ints.	If the specification is not recognized, all
		properties will yield None.
		"""

		self._name = None

		if isinstance(spec, (int, long)):
			self.code = spec
		elif isinstance(spec, basestring):
			try:
				self.code = next(k for k,v in Colorspace.NAME_FROM_CODE.iteritems() if v==spec)
			except StopIteration:
				self.code = None
				self._name = spec
		else:
			raise ValueError("A colorspace spec must be either an integer code or a name")

	# ––––––––––
	# Properties

	@property
	def name(self):
		if self._name is None:
			try:
				return Colorspace.NAME_FROM_CODE[self.code]
			except KeyError:
				return None
		else:
			return self._name

	@property
	def depth(self):
		return Colorspace.DEPTH_FROM_COLORSPACE_NAME[self.name]


	@property
	def _type(self):
		return Colorspace._SAMPLE_TYPE_FROM_COLORSPACE_NAME[self.name]

	# ––––––––––––––––––––
	# Comparison operators

	def __eq__(self, other):
		return (isinstance(other, self.__class__)
			and self.code == other.code)

	def __ne__(self, other):
		return not self.__eq__(other)

	# ––––––––––––––
	# Textualization

	def __str__(self):
		if self.name is not None:
			return self.name
		elif self.code == -1:
			return "Undefined colorspace (-1)"
		else:
			return "Unknown colorspace (code {})".format(self.code)

# ──────────────────────────────── Image class ─────────────────────────────── #

OPERATIONS_AVAILABLE = optional.PySide or optional.cv2
if OPERATIONS_AVAILABLE:
	class Image(object):
		"""
		Image serialization, framework and colorspace conversion facilities.

		Conversion is supported between the ALImage, CVImage and QImage frameworks/
		classes.

		This module depends on the presence of either PySide or cv2, as there would be
		no possibility to convert anything without either, and basic image reading/
		writing or color conversion would be impossible.

		Safe usage must thus be guarded by a test, e.g.:

		import fx.image

		if fx.image.OPERATIONS_AVAILABLE:
			fx.image.Image("Interesting.png")
		"""

		def __init__(self, image = None, image_type = None):
			self.image  = image # Keep a pointer on the object to keep it alive

			if image is None or image_type is not None:
				return

			# The type was unspecified and we have an image object; determine what kind of
			# object and what to do with it
			if Image.isALImage(image):
				self._type = "ALImage"
			elif Image.isCVImage(image):
				# We assume the image is BGR as per OpenCV assumptions
				self._type = "CVImage"
			elif Image.isQImage(image):
				# We assume the image is RGB as per Qt assumptions
				self.image = image.rgbSwapped()
				self._type = "QImage"
			elif Image.isImagePath(image):
				self.load(image)
			else:
				raise RuntimeError("Wrong image type %s" % type(image))

		# ──────────
		# Properties

		@property
		def data(self):
			if self.type == "ALImage":
				return self.image[6]
			elif self.type == "QImage":
				return self.image.bits()
			elif self.type == "CVImage":
				return self.image.data
			else:
				return None

		@property
		def width(self):
			if self.type == "ALImage":
				return self.image[0]
			elif self.type == "QImage":
				return self.image.width()
			elif self.type == "CVImage":
				return self.image.shape[1]
			else:
				return None

		@property
		def height(self):
			if self.type == "ALImage":
				return self.image[1]
			elif self.type == "QImage":
				return self.image.height()
			elif self.type == "CVImage":
				return self.image.shape[0]
			else:
				return None

		@property
		def depth(self):
			"""
			Number of channels.
			"""
			if self.type == "ALImage":
				return self.colorspace.depth
			elif self.type == "QImage":
				# QImages can only contain integers
				return self.image.depth() / 8
			elif self.type == "CVImage":
				if len(self.image.shape) == 3:
					return self.image.shape[2]
				else:
					return 1
			else:
				return None

		@property
		def sampleDepth(self):
			if self.type == "ALImage":
				return self.image[2]
			elif self.type == "QImage":
				return self.image.depth() / 8
			elif self.type == "CVImage":
				if len(self.image.shape) == 3:
					return self.image.shape[2]
				else:
					return 1
			else:
				return None

		@property
		def _sampleType(self):
			"""
			Numpy sample type.
			"""
			if self.type == "ALImage":
				return self.colorspace._type
			elif self.type == "QImage":
				return self.colorspace._type
			elif self.type == "CVImage":
				return self.image.dtype
			else:
				return None
			return self.colorspace._type

		@property
		def resolution(self):
			return Resolution((self.width, self.height))

		@property
		def colorspace(self):
			if self.type == "ALImage":
				return Colorspace(self.image[3])
			elif self.type == "QImage":
				return Colorspace("RGB")
			elif self.type == "CVImage":
				if self.depth == 3:
					return Colorspace("BGR")
				elif self.depth == 1:
					return Colorspace("Gray")
			else:
				return None

		@property
		def type(self):
			assert(any([self._type == t for t in ["ALImage", "QImage", "CVImage"]]))
			return self._type

		# ─────────────────────
		# Conversion Properties

		# Zero-copy conversions, which are thus closer to properties than costly
		# functions

		@property
		def cv_image(self):
			return numpy.frombuffer(self.data, self._sampleType).reshape(self.height,
			                                                             self.width,
			                                                             self.depth)

		@property
		def numpy_image(self):
			return self.cv_image

		@property
		def qimage(self):
			if self.type == "QImage":
				return self

			# Note: QImages can only contain integer images; render them if not integer
			if (self.colorspace not in [Colorspace("RGB"), Colorspace("BGR")]
			 or self._sampleType != numpy.uint8
			 or self.sampleDepth != 3
			 or       self.depth != 3):
				raise ValueError("Can only wrap RGB888 images with QImage")

			rgb_image =  QImage(self.data,
						        self.width, self.height, self.depth*self.width,
						        QImage.Format_RGB888)
			if self.colorspace == Colorspace("RGB"):
				return rgb_image
			elif self.colorspace == Colorspace("BGR"):
				return rgb_image.rgbSwapped()

		# ────────────────────────
		# Image-type determination

		@staticmethod
		def isALImage(o):
			return isinstance(o,list) and len(o)==12

		@staticmethod
		def isCVImage(o):
			return isinstance(o,numpy.ndarray)

		@staticmethod
		def isQImage(o):
			if optional.PySide:
				return isinstance(o, QImage)
			else:
				return False

		@staticmethod
		def isImagePath(o):
			return isinstance(o, basestring) and os.path.isfile(o)

		# ───────────
		# General API

		def render(self):
			if self.colorspace == Colorspace("BGR"):
				return self
			elif not optional.cv2:
				raise RuntimeError("cv2 needed to perform color conversions")
			elif self.colorspace == Colorspace("Yuv"):
				return Image(numpy.dstack((self.numpy_image,self.numpy_image,self.numpy_image)))
			elif self.colorspace == Colorspace("yUv"):
				return Image(numpy.dstack((self.numpy_image,self.numpy_image,self.numpy_image)))
			elif self.colorspace == Colorspace("yuV"):
				return Image(numpy.dstack((self.numpy_image,self.numpy_image,self.numpy_image)))
			elif self.colorspace == Colorspace("Rgb"):
				rgb_image = numpy.zeros((self.height,self.width,3), numpy.uint8)
				rgb_image[:,:,2] = self.numpy_image[:,:,0]
				return Image(rgb_image)
			elif self.colorspace == Colorspace("rGb"):
				rgb_image = numpy.zeros((self.height,self.width,3), numpy.uint8)
				rgb_image[:,:,1] = self.numpy_image[:,:,0]
				return Image(rgb_image)
			elif self.colorspace == Colorspace("rgB"):
				rgb_image = numpy.zeros((self.height,self.width,3), numpy.uint8)
				rgb_image[:,:,0] = self.numpy_image[:,:,0]
				return Image(rgb_image)
			elif self.colorspace == Colorspace("Hsy"):
				return self
			elif self.colorspace == Colorspace("hSy"):
				return self
			elif self.colorspace == Colorspace("hsY"):
				return self
			elif self.colorspace == Colorspace("YUV422"):
				return Image(cv2.cvtColor(self.numpy_image, cv2.COLOR_YUV2BGR_YUYV))
			elif self.colorspace == Colorspace("YUV"):
				return Image(cv2.cvtColor(self.numpy_image, cv2.COLOR_YCrCb2RGB))
			elif self.colorspace == Colorspace("RGB"):
				return Image(cv2.cvtColor(self.numpy_image, cv2.COLOR_RGB2BGR))
			elif self.colorspace == Colorspace("HSY"):
				# TODO Implement but need some HSY documentation
				raise NotImplementedError
			elif self.colorspace == Colorspace("YYCbCr"):
				raw = numpy.ravel(self.numpy_image)
				yuv422 = numpy.empty(raw.shape,raw.dtype)
				# Reorder yyuv into yuyv per 2-pixel/4-byte block
				yuv422[0::4] = raw[0::4] # Y1 🡒 Y1
				yuv422[1::4] = raw[2::4] # Y2 🡖 Cb
				yuv422[2::4] = raw[1::4] # Cb 🡕 Y2
				yuv422[3::4] = raw[3::4] # Cr 🡒 Cr
				return Image(cv2.cvtColor(yuv422.reshape(self.numpy_image.shape), cv2.COLOR_YUV2BGR_YUYV))
			elif self.colorspace == Colorspace("H2RGB"):
				# TODO Implement but need some HSY documentation
				raise NotImplementedError
			elif self.colorspace == Colorspace("HSMixed"):
				# TODO Implement but need some HSY documentation
				raise NotImplementedError
			elif self.colorspace == Colorspace("Depth"):
				return Image.__render1ChannelUint16(self)
			elif self.colorspace == Colorspace("ARGB"):
				# TODO Blind implementation to be checked
				bgr = numpy.empty((self.height,self.width,3),numpy.uint8)
				bgr[:,:,0] = self.numpy_image[:,:,3]
				bgr[:,:,1] = self.numpy_image[:,:,2]
				bgr[:,:,2] = self.numpy_image[:,:,1]
				return Image(bgr)
			elif self.colorspace == Colorspace("XYZ"):
				distances = numpy.linalg.norm(self.numpy_image,axis=2)
				normalizer = distances.max()
				if normalizer == 0.0: normalizer = 1.0
				return Image(distances * 255.0 / normalizer)
			elif self.colorspace == Colorspace("Infrared"):
				return Image.__render1ChannelUint16(self)
			elif self.colorspace == Colorspace("Distance"):
				return Image.__render1ChannelUint16(self)
			elif self.colorspace == Colorspace("Lab"):
				return Image(cv2.cvtColor(self.numpy_image, cv2.COLOR_LAB2BGR))
			elif self.colorspace == Colorspace("RawDepth"):
				return Image.__render1ChannelUint16(self)
			elif self.colorspace == Colorspace("Luv"):
				return Image(cv2.cvtColor(self.numpy_image, cv2.COLOR_Luv2BGR))
			elif self.colorspace == Colorspace("LChab"):
				# TODO Blind implementation to be checked
				L    = self.numpy_image[:,:,0]
				C_ab = self.numpy_image[:,:,1]
				h_ab = self.numpy_image[:,:,2]
				a = numpy.multiply(C_ab, numpy.cos(h_ab))
				b = numpy.multiply(C_ab, numpy.sin(h_ab))
				Lab = numpy.dstack((L,a,b))
				return Image(cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR))
			elif self.colorspace == Colorspace("LChuv"):
				# TODO Blind implementation to be checked
				L    = self.numpy_image[:,:,0]
				C_uv = self.numpy_image[:,:,1]
				h_uv = self.numpy_image[:,:,2]
				Luv = numpy.dstack((L,u,v))
				u = numpy.multiply(C_uv, numpy.cos(h_uv))
				v = numpy.multiply(C_uv, numpy.sin(h_uv))
				return Image(cv2.cvtColor(Luv, cv2.COLOR_Luv2BGR))
			elif self.colorspace == Colorspace("Gray"):
				return self
			else:
				raise RuntimeError("Can't render from colorspace {}".format(self.colorspace))

		@staticmethod
		def __render1ChannelUint16(image):
			assert(isinstance(image,Image))
			assert(image.depth == 1)
			assert(image._sampleType == numpy.uint16)

			normalizer = image.numpy_image.max()
			if normalizer == 0.0: normalizer = 1.0

			# Float64 normalized
			gray_float64_normalized = image.numpy_image * 255.0 / normalizer

			# Gray8 normalized
			# gray_uint8_normalized = numpy.empty((image.height,image.width,1), numpy.uint8)
			# gray_uint8_normalized[:] = image.numpy_image * 255.0 / normalizer

			# Gray16 un-normalized
			# gray_uint16_unnormalized = image.numpy_image

			# TODO Colormap

			return Image(gray_float64_normalized)

		def save(self, path):
			if self.type ==  "CVImage":
				cv2.imwrite(path, self.image)
			elif self.type == "QImage":
				self.image.save(path)
			else:
				if self.colorspace == Colorspace("RGB"):
					if optional.PySide:
						self.qimage.save(path)
					elif optional.cv2:
						cv2.imwrite(path, self.cv_image)
				elif self.colorspace == Colorspace("BGR"):
					if optional.cv2:
						cv2.imwrite(path, self.cv_image)
					elif optional.PySide:
						self.qimage.rgbSwapped().save(path)
				else:
					# Image files have no generic colorspace like ALImage, so we need to convert the
					# image to RGB/BGR before writing it through CVImage or QImage. Additionally
					# color conversions are only implemented by OpenCV, so we can just assume at
					# this point cv2 is available and we'll write with its backend rather than
					# Qt's.

					rendered_image = self.render()
					cv2.imwrite(path, rendered_image.cv_image)

		def load(self, path):
			if optional.cv2:
				self.image = cv2.imread(path)
				self._type = "CVImage"
			elif optional.PySide:
				self.image = QImage(path).rgbSwapped()
				self._type = "QImage"

		# ──────────────
		# Textualization

		def __str__(self):
			colorspace_string = str(self.colorspace)
			if self.type in ["CVImage", "QImage"]:
				colorspace_string += " (by " + self.type + " convention)"
			return "{}✕{}✕{} image at {} in {}".format(self.width, self.height, self.depth,
													   hex(id(self.data)), colorspace_string)

	def similarCVImage(cv_image):
		return numpy.empty(cv_image.shape, cv_image.dtype)

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
