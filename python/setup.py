import os
import re
import sys
import shutil
import platform
import subprocess

from setuptools import find_packages
from setuptools import setup, Extension
from setuptools.dist import Distribution
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

CURRENT_DIR = os.path.dirname(__file__)

def get_lib_path():
    """Get library path, name and version"""
     # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(CURRENT_DIR, 'tfdlpack', 'libinfo.py')
    libinfo = {'__file__': libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)
    version = libinfo['__version__']

    lib_path = libinfo['find_lib_path']()
    libs = [lib_path[0]]

    return libs, version

LIBS, VERSION = get_lib_path()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            raise RuntimeError("Windows not currently supported")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        extdir = os.path.join(extdir, "tfdlpack", "build") # Not sure whether this is fine

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        env = os.environ.copy()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] +
                              build_args, cwd=self.build_temp)

include_libs = False
wheel_include_libs = False
if "bdist_wheel" in sys.argv or os.getenv('CONDA_BUILD'):
    wheel_include_libs = True
else:
    include_libs = True

setup_kwargs = {}

# For bdist_wheel only
if wheel_include_libs:
    with open("MANIFEST.in", "w") as fo:
        for path in LIBS:
            shutil.copy(path, os.path.join(CURRENT_DIR, 'tfdlpack'))
            _, libname = os.path.split(path)
            fo.write("include tfdlpack/%s\n" % libname)
    setup_kwargs = {
        "include_package_data": True
    }

# For source tree setup
# Conda build also includes the binary library
if include_libs:
    rpath = [os.path.relpath(path, CURRENT_DIR) for path in LIBS]
    setup_kwargs = {
        "include_package_data": True,
        "data_files": [('tfdlpack', rpath)]
    }

setup(
    name='tfdlpack' + os.getenv('TFDLPACK_PACKAGE_SUFFIX', ''),
    version=VERSION,
    author='Jinjing Zhou',
    author_email='allen.zhou@nyu.edu',
    description='Tensorflow plugin for DLPack',
    packages=find_packages(),
    install_requires=['tensorflow%s>=2.0.0' % os.getenv('TFDLPACK_PACKAGE_SUFFIX')],
    long_description="""
The package adds interoperability of DLPack to Tensorflow. It contains straightforward
and easy-to-use APIs to convert Tensorflow tensors from/to DLPack format.
    """,
    distclass=BinaryDistribution,
    zip_safe=False,
    license='APACHE',
    **setup_kwargs
)

if wheel_include_libs:
    # Wheel cleanup
    os.remove("MANIFEST.in")
    for path in LIBS:
        _, libname = os.path.split(path)
        os.remove(os.path.join(CURRENT_DIR, 'tfdlpack', libname))
