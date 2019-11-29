#!/bin/bash
TF=$1

wget -O /tmp/install.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh /tmp/install.sh -b

CONDA_PREFIX=$HOME/miniconda3/bin
export PATH=$CONDA_PREFIX:$PATH
for PY_VER in 3.6.4 3.7.0; do
  echo "Create conda env for python $PY_VER"
  conda create -n $PY_VER -y python=$PY_VER
  source activate $PY_VER
  conda install -y $TF==2.0
  source deactivate
done
