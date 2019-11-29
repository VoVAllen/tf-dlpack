#!/bin/bash

set -e

if [ -d build ]; then
  rm -rf build
fi

mkdir build

pushd build
cmake ..
make -j4
popd

export TFDLPACK_LIBRARY_PATH=$PWD/build
export TFDLPACK_PACKAGE_SUFFIX=-gpu

pushd python
python3 setup.py clean
python3 setup.py install
popd
