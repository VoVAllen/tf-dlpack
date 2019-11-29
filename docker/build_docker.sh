#/usr/bin/sh
# From current directory
docker build -t dgllib/tfdlpack-ci-cpu -f Dockerfile.cpu .
docker build -t dgllib/tfdlpack-ci-gpu -f Dockerfile.gpu .
