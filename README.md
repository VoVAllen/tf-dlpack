# DLPack for Tensorflow
[![Build Status](http://ci.dgl.ai:80/buildStatus/icon?job=tf-dlpack/master)](http://ci.dgl.ai:80/job/tf-dlpack/job/master/) 

Notes: Currently only tested under tensorflow 2.0's eager mode. Implementation details could be found [here](https://github.com/VoVAllen/tf-dlpack/issues/3).


## Install
Set allow growth, otherwise tf would take over whole gpu
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

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
cmake ..  # To build without CUDA, add -DUSE_CUDA=OFF
make -j4
```

Export the library path:
```bash
export TF_DLPACK_LIBRARY_PATH=/path/to/tf-dlpack/repo/build
```

Export python path to `import tfdlpack`
```bash
export PYTHONPATH=/path/to/tf-dlpack/repo/python/:${PYTHONPATH}
```

## License

[Apache License 2.0](LICENSE)
