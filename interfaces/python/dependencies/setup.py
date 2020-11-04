#! /usr/bin/env python

import distutils.core
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

_conex = distutils.core.Extension("_conex",
                   ["conex.i"],
                   libraries = ["/home/frank/conexnew/conex/interfaces/conex"],
                   include_dirs = [numpy_include],
                   )

distutils.core.setup(name        = "Conex",
                  description = "Provides python interface to the Conex optimizer",
                  author      = "Frank Permenter",
                  version     = "1.0",
                  ext_modules = [_conex])

