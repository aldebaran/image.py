# Import main class
from .image import Image, Colorspace

# Define package version
import os as _os
__VERSION__=open(
    _os.path.join(
        _os.path.dirname(_os.path.realpath(__file__)),
        "VERSION"
    )
).read().split()[0]
