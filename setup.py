from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["protovoc/numericalization/cython/*.pyx", "protovoc/vocab/cython.pyx"]),
    include_dirs=[np.get_include()]
)
