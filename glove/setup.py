from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
from os import path

ext_modules = [
    Extension("glove.glove_inner", ["glove/*.pyx"], include_dirs=[np.get_include()])
]


def readfile(fname):
    return open(path.join(path.dirname(__file__), fname)).read()


setup(
    name="glove",
    version="1.0.0",
    description=("Vectorized implementation of GloVe embedding method."),
    long_description=readfile("README.md"),
    py_modules=[],
    author="Jonathan Raiman",
    author_email="jraiman at mit dot edu",
    url="https://github.com/JonathanRaiman/glove",
    download_url="https://github.com/JonathanRaiman/glove",
    keywords="NLP, Machine Learning",
    license="MIT",
    platforms="any",
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
    setup_requires=[],
    include_package_data=True,
    license="MIT",
    python_requires=">=3.8",
    install_requires=["Cython", "numpy>=1.19", "scipy>=1.6.3"],
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, language_level="3"),
)
