# DLPack for Tensorflow
[![Build Status](http://ci.dgl.ai:80/buildStatus/icon?job=tf-dlpack/master)](http://ci.dgl.ai:80/job/tf-dlpack/job/master/) 

Notes: Currently only tested under tensorflow 2.0's eager mode. Implementation details could be found [here](https://github.com/VoVAllen/tf-dlpack/issues/3).

## Install

Pip install
```bash
pip install tfdlpack  # no cuda
# pip install tfdlpack-gpu  # with cuda support
```

## Usage
Set allow growth, otherwise tf would take over whole gpu
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

Use `tfdlpack`

```python
import tfdlpack
dl_capsule = tfdlpack.to_dlpack(tf_tensor)    # Convert tf tensor to dlpack capsule
tf_tensor = tfdlpack.from_dlpack(dl_capsule)  # Convert dlpack capsule to tf tensor
```

## Build and develop locally

Build plugin library
```
mkdir build
cd build
cmake ..  # To build without CUDA, add -DUSE_CUDA=OFF
make -j4
```

Export the library path:
```bash
export TFDLPACK_LIBRARY_PATH=/path/to/tf-dlpack/repo/build
```

Export python path to `import tfdlpack`
```bash
export PYTHONPATH=/path/to/tf-dlpack/repo/python/:${PYTHONPATH}
```


## License

[Apache License 2.0](LICENSE)
