#!/bin/bash

set -e

BRANCH=$1

rm -rf cpu-release gpu-release

mkdir cpu-release
docker run -it --rm -v "$PWD":/workspace \
  --name tfdlpack-build-cpu dgllib/tfdlpack-ci-cpu:latest \
  bash /workspace/build_in_docker.sh $BRANCH OFF
mv *.whl cpu-release

mkdir gpu-release
docker run -it --rm -v "$PWD":/workspace \
  --runtime nvidia \
  --name tfdlpack-build-gpu dgllib/tfdlpack-ci-gpu:latest \
  bash /workspace/build_in_docker.sh $BRANCH ON
mv *.whl gpu-release
