#!/bin/bash
DEV=$1

if [ $DEV = "cpu" ]; then
  TF="tensorflow"
  TH="pytorch cpuonly"
else
  TF="tensorflow-gpu"
  TH="pytorch"
fi

wget -O /tmp/install.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh /tmp/install.sh -b

CONDA_PREFIX=$HOME/miniconda3/bin
export PATH=$CONDA_PREFIX:$PATH
for PY_VER in 3.5 3.6 3.7; do
  echo "Create conda env for python $PY_VER"
  conda create -n $PY_VER -y python=$PY_VER
  source activate $PY_VER
  pip install --upgrade pip
  pip install $TF==2.1.0
  conda install -y pytest
  conda install -y $TH -c pytorch
  source deactivate
done
