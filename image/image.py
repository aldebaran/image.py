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

# Standard libraries
import collections
import os

# Third-party libraries
try:
	import cv2
	_has_CV = True
except ImportError:
	_has_CV = False

from enum import Enum, EnumMeta
import numpy as numpy
try:
	from Qt.QtGui import QPixmap, QImage
	_has_Qt = True
except ImportError:
	_has_Qt = False

# ──────────────────────── Embedded vision definitions ─────────────────────── #

class ColorspaceMeta(EnumMeta):
	def __call__(
	    cls, value=None, names=None, module=None, type=None, start=1,
	    al_code=None, qt_code=None):
		if al_code is not None\
		   or qt_code is not None:
			return cls.__new__(cls, al_code=al_code, qt_code=qt_code)
		if not value:
			raise TypeError("__call__() requires at least a 'value' argument")
		return EnumMeta.__call__(cls, value, names, module, type, start)

class Colorspace(Enum):
	__metaclass__ = ColorspaceMeta

	# –––––––––––––––
	# Enum definition

	# Name | AL Code | Qt code | # of channels | Inner type

	Yuv      = (   0,   24, 1, numpy.uint8)
	Gray     = (   0,   24, 1, numpy.uint8) # alias for Yuv
	yUv      = (   1, None, 1, numpy.uint8)
	yuV      = (   2, None, 1, numpy.uint8)
	Rgb      = (   3, None, 1, numpy.uint8)
	rGb      = (   4, None, 1, numpy.uint8)
	rgB      = (   5, None, 1, numpy.uint8)
	Hsy      = (   6, None, 1, numpy.uint8)
	hSy      = (   7, None, 1, numpy.uint8)
	hsY      = (   8, None, 1, numpy.uint8)
	YUV422   = (   9, None, 3, numpy.uint8)
	YUV      = (  10, None, 3, numpy.uint8)
	RGB      = (  11,   13, 3, numpy.uint8) # 24-bit RGB format (8-8-8)
	RGB32    = (None,    4, 4, numpy.uint8) # 32-bit RGB format (0xffRRGGBB).
	HSY      = (  12, None, 3, numpy.uint8)
	BGR      = (  13, None, 3, numpy.uint8)
	YYCbCr   = (  14, None, 2, numpy.uint8)
	H2RGB    = (  15, None, 3, numpy.uint8)
	HSMixed  = (  16, None, 3, numpy.uint8)
	Depth    = (  17, None, 1, numpy.uint16)
	ARGB     = (  18,    5, 4, numpy.uint8) # 32-bit ARGB format (0xAARRGGBB)
	XYZ      = (  19, None, 3, numpy.float32)
	Infrared = (  20, None, 1, numpy.uint16)
	Distance = (  21, None, 1, numpy.uint16)
	Lab      = (  22, None, 3, numpy.uint8)
	RawDepth = (  23, None, 1, numpy.uint16)
	Luv      = (  24, None, 3, numpy.uint8)
	LChab    = (  25, None, 3, numpy.uint8)
	LChuv    = (  26, None, 3, numpy.uint8)

	## Yet unsupported types from Qt

	# Format_Invalid                 0
	# Format_Mono                    1
	# Format_MonoLSB                 2
	# Format_Indexed8                3
	# Format_ARGB32_Premultiplied    6
	# Format_RGB16                   7

	## From Qt 4.4

	# Format_ARGB8565_Premultiplied	 8
	# Format_RGB666	                 9
	# Format_ARGB6666_Premultiplied	10
	# Format_RGB555	                11
	# Format_ARGB8555_Premultiplied	12
	# Format_RGB444	                14
	# Format_ARGB4444_Premultiplied	15

	## From Qt 5.2

	# Format_RGBX8888               16
	# Format_RGBA8888               17
	# Format_RGBA8888_Premultiplied	18

	## From Qt 5.4

	# Format_BGR30                  19
	# Format_A2BGR30_Premultiplied	20
	# Format_RGB30                  21
	# Format_A2RGB30_Premultiplied	22

	## From Qt 5.5

	# Format_Alpha8                 23


	# COLOR    = set(range(17)+[18,22,24,25,26])
	# DEPTH    = set([17,19,21,23])
	# INFRARED = set([20])

	# ––––––––––––––
	# Static Helpers

	@staticmethod
	def allKnownNames():
		return list(Colorspace)

	# –––––––––––
	# Constructor

	@staticmethod
	def __another_new__(cls, value=None, al_code=None, qt_code=None):
		"""
		Will override the __new__ method of Colorspace to customize colorspace
		search based on color code

		__new__ cannot be directly overriden in the class itself due to the way
		Enum metaclass works (so sad..). So we need to use the default new first
		and then override it after class construction
		"""
		# all enum instances are actually created during class construction
		# without calling this method; this method is called by the metaclass'
		# __call__ (i.e. Color(3) ), and by pickle
		if value is not None:
			if type(value) is cls:
				# For lookups like Colorspace(Colorspace.RGB)
				# Retrieve the defining tuple
				value = value.value

			# Look for the exact value (the tuple defining the colorspace)
			if value in cls._value2member_map_:
				return cls._value2member_map_[value]

			# Not found, try to use value as a colorspace name
			try:
				return cls[value]
			except ValueError:
				pass

		if al_code is not None:
			# Search colorspace by ALImage codes
			for colorspace_details in cls._value2member_map_:
				if colorspace_details[0] == al_code:
					return cls._value2member_map_[colorspace_details]

		if qt_code is not None:
			# Search colorspace by ALImage codes
			for colorspace_details in cls._value2member_map_:
				if colorspace_details[1] == qt_code:
					return cls._value2member_map_[colorspace_details]

		raise ValueError("%s is not a valid %s" % (value, cls.__name__))

	# ––––––––––
	# Properties

	@property
	def al_code(self):
		return self.value[0]

	@property
	def qt_code(self):
		return self.value[1]

	@property
	def depth(self):
		return self.value[2]

	@property
	def dtype(self):
		return self.value[3]

	# ––––––––––––––
	# Textualization

	def __str__(self):
		return self.name

setattr(Colorspace, '__new__', Colorspace.__dict__["__another_new__"])

# ──────────────────────────────── Image class ─────────────────────────────── #

class Image(object):
	"""
	Image serialization, framework and colorspace conversion facilities.

	Conversion is supported between the ALImage, CVImage and QImage frameworks/
	classes.

	This module depends on the presence of either PySide or cv2, as there would be
	no possibility to convert anything without either, and basic image reading/
	writing or color conversion would be impossible.
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
		if hasattr(self, "_colorspace"):
			return self._colorspace

		# Colorspace is unknown, make a guess based on the inner data
		if numpy.float32 == self._sampleType:
			# Only one option
			return Colorspace("XYZ")

		elif numpy.uint16 == self._sampleType:
			# Possible options:
			# - Depth (*)
			# - Infrared
			# - Distance
			# - RawDepth
			return Colorspace("Depth")

		elif self.depth == 1:
			# Possible options:
			# - Yuv (*)
			# - yUv
			# - yuV
			# - Rgb
			# - rGb
			# - rgB
			# - Hsy
			# - hSy
			# - hsY
			return Colorspace("Yuv")

		elif self.depth == 2:
			# Only one option
			return Colorspace("YYCbCr")

		elif self.depth == 3:
			# Possible options:
			# - YUV422
			# - YUV
			# - RGB
			# - HSY
			# - BGR (*)
			# - H2RGB
			# - HSMixed
			# - Lab
			# - Luv
			# - LChab
			# - LChuv
			return Colorspace("BGR")

		elif self.depth == 4:
			# Possible options
			# - ARGB (*)
			# - RGB32
			return Colorspace("ARGB")

	@colorspace.setter
	def colorspace(self, colorspace):
		self._colorspace = Colorspace(colorspace)

	@property
	def type(self):
		assert(any([self._type == t for t in ["ALImage", "QImage", "CVImage"]]))
		return self._type

	# ─────────────────────
	# Conversion Properties

	# Zero-copy conversions, which are thus closer to properties than costly
	# functions

	@property
	def numpy_image(self):
		return numpy.frombuffer(self.data, self._sampleType).reshape(self.height,
		                                                             self.width,
		                                                             self.depth)

	if _has_CV:
		@property
		def cv_image(self):
			return self.numpy_image

	if _has_Qt:
		@property
		def qimage(self):
			if self.type == "QImage":
				return self

			# Note: QImages can only contain integer images; render them if not integer
			if (self.colorspace not in [Colorspace("BGR"), Colorspace("RGB")]
			 or self._sampleType != numpy.uint8
			 or self.sampleDepth != 3
			 or       self.depth != 3):
				return self.render().qimage

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
		if _has_Qt:
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
		elif not _has_CV:
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
			return Image(numpy.dstack((self.numpy_image,self.numpy_image,self.numpy_image)))
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

		return Image(numpy.dstack((gray_uint8_normalized,gray_uint8_normalized,gray_uint8_normalized)))

	def save(self, path):
		if self.type ==  "CVImage":
			cv2.imwrite(path, self.image)
		elif self.type == "QImage":
			self.image.save(path)
		else:
			if self.colorspace == Colorspace("RGB"):
				if _has_Qt:
					self.qimage.save(path)
				elif _has_CV:
					cv2.imwrite(path, self.cv_image)
			elif self.colorspace == Colorspace("BGR"):
				if _has_CV:
					cv2.imwrite(path, self.cv_image)
				elif _has_Qt:
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
		if _has_CV:
			self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
			self._type = "CVImage"
		elif _has_Qt:
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
