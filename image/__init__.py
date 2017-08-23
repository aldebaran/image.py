# Import main class
from .image import Image

# Define package version
import os as _os
__VERSION__=open(
    _os.path.join(
        _os.path.dirname(_os.path.realpath(__file__)),
        "image/VERSION"
    )
).read().split()[0]
