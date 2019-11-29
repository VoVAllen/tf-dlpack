#/bin/bash

set -e

BRANCH=$1
USE_CUDA=$2

pushd /tmp
git clone https://github.com/VoVAllen/tf-dlpack.git -b $BRANCH
pushd tf-dlpack

CONDA_PREFIX=$HOME/miniconda3/bin
export PATH=$CONDA_PREFIX:$PATH
export PYTHONPATH=$PWD/python:$PYTHONPATH
export TF_DLPACK_LIBRARAY_PATH=$PWD/build
for PY_VER in 3.6.4 3.7.0; do
  echo "Build for python $PY_VER"
  source activate $PY_VER
  # clean & build
  rm -rf build
  mkdir build
  cd build; cmake -DUSE_CUDA=$USE_CUDA ..; make -j; cd ..
  python setup.py clean
  python setup.py bdist_wheel --plat-name manylinux1_x86_64
  source deactivate
done

ls -l dist

popd
popd
