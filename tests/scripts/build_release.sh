#!/bin/bash

set -e

BRANCH=$1

rm -rf *.whl

docker pull dgllib/tfdlpack-ci-cpu:03302020

docker run -it --rm -v "$PWD":/workspace \
  --name tfdlpack-build-cpu dgllib/tfdlpack-ci-cpu:latest \
  bash /workspace/build_in_docker.sh $BRANCH OFF

docker pull dgllib/tfdlpack-ci-gpu:03302020

docker run -it --rm -v "$PWD":/workspace \
  --runtime nvidia \
  --name tfdlpack-build-gpu dgllib/tfdlpack-ci-gpu:latest \
  bash /workspace/build_in_docker.sh $BRANCH ON
