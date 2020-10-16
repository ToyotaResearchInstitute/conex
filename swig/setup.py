#! /usr/bin/env python
import re
import requests
import numpy


# System imports
from distutils.core import *
from distutils      import sysconfig



np_version = re.compile(r'(?P<MAJOR>[0-9]+)\.'
                        '(?P<MINOR>[0-9]+)') \
                        .search(numpy.__version__)
np_version_string = np_version.group()
np_version_info = {key: int(value)
                   for key, value in np_version.groupdict().items()}

np_file_name = 'numpy.i'
np_file_url = 'https://raw.githubusercontent.com/numpy/numpy/maintenance/' + \
              np_version_string + '.x/tools/swig/' + np_file_name
if(np_version_info['MAJOR'] == 1 and np_version_info['MINOR'] < 9):
    np_file_url = np_file_url.replace('tools', 'doc')

chunk_size = 8196
with open(np_file_name, 'wb') as file:
    for chunk in requests.get(np_file_url,
                              stream=True).iter_content(chunk_size):
        file.write(chunk)

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

_conex = Extension("_conex",
                   ["conex.i"],
                   libraries = ["conex"],
                   include_dirs = [numpy_include],
                   library_dirs = ["/home/frank/research/conex/swig"],
                   )

setup(name        = "Conex",
      description = "Provides python interface to the Conex optimizer",
      author      = "Frank Permenter",
      version     = "1.0",
      ext_modules = [_conex])

