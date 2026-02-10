from Cython.Build import cythonize

from setuptools import setup, find_packages, Extension

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    ext_modules=cythonize(
        [
            Extension("toolkit.maths.c.angles", ["src/toolkit/maths/c/angles.pyx"]),
            Extension("toolkit.maths.c.normals", ["src/toolkit/maths/c/normals.pyx"]),
            Extension("toolkit.maths.c.lines", ["src/toolkit/maths/c/lines.pyx"]),
            Extension("toolkit.maths.c.points", ["src/toolkit/maths/c/points.pyx"]),
            Extension("toolkit.maths.c.splines", ["src/toolkit/maths/c/splines.pyx"]),
            Extension("toolkit.maths.c.intersections", ["src/toolkit/maths/c/intersections.pyx"]),
            Extension("toolkit.maths.c.functional", ["src/toolkit/maths/c/functional.pyx"]),
        ],
        language_level="3",
        annotate=True,

    )
)


# export ARCHFLAGS="-arch x86_64" && python setup.py build_ext --inplace
