"""Setup module for Andor SDK2.

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import distutils.sysconfig

from codecs import open
from os import path

import glob
import sys
import platform
import shutil, errno

here = path.abspath(path.dirname(__file__))
_package_name = 'pyAndorSDK2'
relative_directory = "../docs/"
def get_max_bits():
    if (sys.maxsize > 2**32):
        return '64'
    else:
        return '32'
        
        
def get_platform_name():
    return platform.system()
    
    
def get_lib_wildcard():
    if (get_platform_name() == 'Windows'):
        return '*.dll'
    else:
        return '*.so*'

with open(path.join(here, relative_directory , 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

version_ns = {}
with open(path.join(here, _package_name, '_version.py')) as f:
    exec(f.read(), {}, version_ns)
    
setup(
    name=_package_name,
    version=version_ns['__version__'],
    description='Provides a wrapper for the Andor SDK2 API',
    long_description=long_description,
    url='http://my.andor.com/user/',
    author='Andor CCD team',
    author_email='row_productsupport@andor.com',
    license='Andor internal',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Andor internal',
        'Programming Language :: Python :: 3.6',
    ],

    packages=[_package_name],
    install_requires=['cffi', 'numpy', 'matplotlib', 'astropy'],
    zip_safe=False
)

# Install platform specific libraries into standard library directory
from distutils import dir_util
from distutils import sysconfig

destination_path = sysconfig.get_python_lib()
source_lib_path = path.join(here, _package_name, 'libs', get_platform_name(), get_max_bits())
package_lib_path = path.join(destination_path, _package_name, 'libs')

dir_util.copy_tree(source_lib_path, package_lib_path, update=1, preserve_mode=0)
