# DLPack for TF

Set allow growth, otherwise tf would take over whole gpu
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

Build
```
mkdir build
cd build
cmake ..
make -j4
```

so file path is now fixed in `python/tfdlpack/__init__.py`
Need to change manually

And export the python path to `import tfdlpack`
```bash
export PYTHONPATH=/home/ubuntu/dev/tfdlpack/python/:${PYTHONPATH}
```