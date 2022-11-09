from setuptools import setup, Extension
import numpy as np

setup(
    name='newton_method',
    version='0.1.0',
    author='yuruto',
    ext_modules=[
        Extension(
            name='wrapper_newtonlib',
            sources=[
                'libs/src/wrapper.c',
                'libs/src/newton_method.c',
            ],
            include_dirs=['./libs/include', np.get_include()]
        ),
    ],
    zip_safe=False,
)
