# DLPack for TF
[![Build Status](http://ci.dgl.ai:80/buildStatus/icon?job=tf-dlpack/master)](http://ci.dgl.ai:80/job/tf-dlpack/job/master/)

Set allow growth, otherwise tf would take over whole gpu
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Install

### Pip install
```bash
pip install git+https://github.com/VoVAllen/tf-dlpack.git
```

### Local install
```bash
python setup.py install
# or
pip install .
```

## Usage
```python
import tfdlpack
dl_capsule = tfdlpack.to_dlpack(tf_tensor)    # Convert tf tensor to dlpack capsule
tf_tensor = tfdlpack.from_dlpack(dl_capsule)  # Convert dlpack capsule to tf tensor
```


## Build Manually

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

## License

[Apache License 2.0](LICENSE)
