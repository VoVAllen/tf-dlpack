"""Library information."""
from __future__ import absolute_import
import sys
import os


def find_lib_path(name=None, search_path=None, optional=False):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    # See https://github.com/dmlc/tvm/issues/281 for some background.

    # NB: This will either be the source directory (if tfdlpack is run
    # inplace) or the install directory (if tfdlpack is installed).
    # An installed tfdlpack's curr_path will look something like:
    #   $PREFIX/lib/python3.6/site-packages/tfdlpack
    source_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    install_lib_dir = os.path.join(source_dir, "build")

    dll_path = []

    if os.environ.get('TFDLPACK_LIBRARY_PATH', None):
        dll_path.append(os.environ['TFDLPACK_LIBRARY_PATH'])

    if sys.platform.startswith('linux') and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")])
    elif sys.platform.startswith('darwin') and os.environ.get('DYLD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['DYLD_LIBRARY_PATH'].split(":")])

    # Pip lib directory
    dll_path.append(source_dir)
    # Default cmake build directory
    dll_path.append(os.path.join(source_dir, "build"))
    dll_path.append(os.path.join(source_dir, "build", "Release"))
    # Default make build directory
    dll_path.append(os.path.join(source_dir, "lib"))

    dll_path.append(install_lib_dir)

    dll_path = [os.path.abspath(x) for x in dll_path]
    if search_path is not None:
        if search_path is list:
            dll_path = dll_path + search_path
        else:
            dll_path.append(search_path)
    if name is not None:
        if isinstance(name, list):
            lib_dll_path = []
            for n in name:
                lib_dll_path += [os.path.join(p, n) for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, name) for p in dll_path]
    else:
        if sys.platform.startswith('win32'):
            lib_dll_path = [os.path.join(p, 'libtfdlpack.dll') for p in dll_path] +\
                           [os.path.join(p, 'tfdlpack.dll') for p in dll_path]
        elif sys.platform.startswith('darwin'):
            lib_dll_path = [os.path.join(p, 'libtfdlpack.dylib') for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, 'libtfdlpack.so') for p in dll_path]

    # try to find lib_dll_path
    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_found:
        message = ('Cannot find the files.\n' +
                   'List of candidates:\n' +
                   str('\n'.join(lib_dll_path)))
        if not optional:
            raise RuntimeError(message)
        return None

    return lib_found


# current version
# We use the version of the incoming release for code
# that is under development.
# The following line is set by tfdlpack/python/update_version.py
__version__ = "0.1"
