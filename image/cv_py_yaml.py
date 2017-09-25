# Standard libraries
import warnings

# Third-party libraries
import numpy
import yaml

def openCvYamlFile(cv_yaml_file):
	"""
	Open a yaml file written by OpenCv, and turn it into readable yaml file.

	:param cv_yaml_file: path to the opencv yaml file
	:type cv_yaml_file: str
	:returns: yaml file content without anything causing problems to PyYaml
	:rtype: str
	"""
	with open(cv_yaml_file, "r") as stream:
		lines = stream.readlines()
		if "YAML" in lines[0]:
			# Remove commented lines because %YAML:1.0 causes troubles with python yaml.
			return "".join([l for l in lines if l[0] != "%"])
		else:
			raise Exception("File format not supported")

def writeCvYamlFile(cv_yaml_file, content):
	"""
	Write a content to a yaml file using OpenCv standard.

	:param cv_yaml_file: path to the opencv yaml file to write
	:type cv_yaml_file: str
	:param content: content to write
	:type content: str
	"""
	with open(cv_yaml_file, "w") as stream:
		stream.writelines(["%%YAML:1.0\n"])
		stream.write(content)


# A yaml constructor is for loading from a yaml node.
# This is taken from @misha 's answer: http://stackoverflow.com/a/15942429
# and modified
def _opencvNDMatrixConstructor(loader, node):
	mapping = loader.construct_mapping(node, deep=True)
	def icvDecodeFormat(type_symbol):
		if type_symbol == "u":
			return numpy.uint8
		elif type_symbol == "c":
			return numpy.int8
		elif type_symbol == "w":
			return numpy.uint16
		elif type_symbol == "s":
			return numpy.int16
		elif type_symbol == "i":
			return numpy.int32
		elif type_symbol == "f":
			return numpy.float32
		elif type_symbol == "d":
			return numpy.float64
		else:
			warnings.warn("YAML-to-CV: Unknown type symbol %s, trying to infer it from data"%type_symbol)
			return None

	mat = numpy.array(mapping["data"], dtype=icvDecodeFormat(mapping.get("dt",None)))

	if mapping.has_key("sizes"):
		mat.resize(mapping["sizes"])
	else:
		mat.resize(mapping["rows"], mapping["cols"])
	return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-nd-matrix", _opencvNDMatrixConstructor)
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", _opencvNDMatrixConstructor)

# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def _opencvMatrixRepresenter(dumper, mat):

	def icvEncodeFormat(numpy_type):
		if numpy_type == numpy.uint8:
			return "u"
		elif numpy_type == numpy.int8:
			return "c"
		elif numpy_type == numpy.uint16:
			return "w"
		elif numpy_type == numpy.int16:
			return "s"
		elif numpy_type == numpy.int32:
			return "i"
		elif numpy_type == numpy.float32:
			return "f"
		elif numpy_type == numpy.float64:
			return "d"
		else:
			raise TypeError("Not an OpenCV compatible-type: %s"%numpy_type)

	if len(mat.shape)>2:
		mapping = {'sizes': list(mat.shape), 'dt': icvEncodeFormat(mat.dtype), 'data': mat.reshape(-1).tolist()}
		return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-nd-matrix", mapping)
	else:
		mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': icvEncodeFormat(mat.dtype), 'data': mat.reshape(-1).tolist()}
		return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)
yaml.add_representer(numpy.ndarray, _opencvMatrixRepresenter)


def _loadCalibrationFile(calibration_file):
	"""
	Load a calibration file in YAML format, typically returned by the aruco
	sample programs.
	"""
	data = yaml.load(openCvYamlFile(calibration_file))

	# Check mandatory data
	if not "width" in data.keys()\
	   or not "height" in data.keys()\
	   or not "cameraMatrix" in data.keys()\
	   or not "distortion" in data.keys()\
	   or not "rectificationMatrix" in data.keys()\
	   or not "projectionMatrix" in data.keys():
		raise KeyError("All mandatory keys are not present in calibration file")

	if not "calibration_time" in data.keys():
		warnings.warn("No calibration time, this could be a default calibration file", RuntimeWarning)

	return data
