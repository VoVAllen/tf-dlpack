#/bin/bash

set -e

BRANCH=$1
USE_CUDA=$2

pushd /tmp
git clone https://github.com/VoVAllen/tf-dlpack.git --recursive
pushd tf-dlpack
git checkout $BRANCH

CONDA_PREFIX=$HOME/miniconda3/bin
export PATH=$CONDA_PREFIX:$PATH
export PYTHONPATH=$PWD/python:$PYTHONPATH
export TFDLPACK_LIBRARY_PATH=$PWD/libs
for PY_VER in 3.5 3.6 3.7; do
  echo "Build for python $PY_VER"
  source activate $PY_VER
  # clean & build
  rm -rf libs
  mkdir libs
  for TF_VER in "2.1.0" "2.2.0rc2"; do
    pip uninstall -y tensorflow
    if [ $PY_VER = "3.5" ]; then
      pip uninstall -y numpy
      pip install numpy
    fi
    pip install $HOME/$PY_VER/tensorflow*${TF_VER}*.whl
    rm -rf build
    mkdir build
    cd build; cmake -DUSE_CUDA=$USE_CUDA ..; make -j; cd ..
    mv build/*.so libs
  done
  # test
  if [ $USE_CUDA = "ON" ]; then
    python -m pytest tests
    export TFDLPACK_PACKAGE_SUFFIX=-gpu
  else
    export TFDLPACK_PACKAGE_SUFFIX=
  fi
  # build wheel
  pushd python
  python setup.py clean
  python setup.py bdist_wheel --plat-name manylinux1_x86_64
  popd
  source deactivate
done

cp python/dist/*.whl /workspace

popd
popd
