__title__ = "pyAndorSDK2"

__authors__ = "Andor SDK2 team"
__email__ = "row_productsupport@andor.com"

__license__ = "Andor internal"
__copyright__ = "Copyright 2017 Andor"

import os
import sys

_path = os.path.dirname(__file__) + "/libs;" + os.environ["PATH"]
os.environ["PATH"] = _path

# from pyAndorSDK2._version import __version__, __version_info__

# __all__ = [
#     '__title__', '__authors__', '__email__', '__license__',
#     '__copyright__', '__version__', '__version_info__',
# ]
