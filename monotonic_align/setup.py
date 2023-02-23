from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Check the following links to learn why I modify the code
# https://stackoverflow.com/questions/71983019/why-does-adding-an-init-py-change-cython-build-ext-inplace-behavior
# https://github.com/python/cpython/issues/93347
package = Extension('core', ['core.pyx'], include_dirs=[numpy.get_include()])
setup(
  ext_modules = cythonize(package)
)