# DLPack for Tensorflow
[![Build Status](http://ci.dgl.ai:80/buildStatus/icon?job=tf-dlpack/master)](http://ci.dgl.ai:80/job/tf-dlpack/job/master/) 

Notes: Currently only tested under tensorflow 2.0's eager mode. Implementation details could be found [here](https://github.com/VoVAllen/tf-dlpack/issues/3).

## Install

Pip install
```bash
pip install tfdlpack  # no cuda
pip install tfdlpack-gpu  # with cuda support
```

## Quick Start

Use `tfdlpack`

```python
import tfdlpack
dl_capsule = tfdlpack.to_dlpack(tf_tensor)    # Convert tf tensor to dlpack capsule
tf_tensor = tfdlpack.from_dlpack(dl_capsule)  # Convert dlpack capsule to tf tensor
```

## Usage
Set allow growth, otherwise TensorFlow would take over the whole GPU
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### [CuPy](https://github.com/cupy/cupy) <-> TensorFlow
```python
# pip install cupy>=6.2.0
import cupy as cp

# CuPy - GPU Array (like NumPy!)
gpu_arr = cp.random.rand(10_000, 10_000)

# Use CuPy's built in `toDlpack` function to move to a DLPack capsule
dlpack_arr = gpu_arr.toDlpack()

# Use `tfdlpack` to migrate to TensorFlow
tf_tensor = tfdlpack.from_dlpack(dlpack_arr)

# Confirm TF tensor is on GPU
print(tf_tensor.device)

# Use `tfdlpack` to migrate back to CuPy
dlpack_capsule = tfdlpack.to_dlpack(tf_tensor)
cupy_arr = cp.fromDlpack(dlpack_capsule)
```

### [Numba](https://github.com/numba/numba) CUDA <-> TensorFlow
```python
# pip install numba numpy
import numpy as np
from numba import cuda

# NumPy - CPU Array
cpu_arr = np.random.rand(10_000, 10_000)

# Use Numba to move to GPU
numba_gpu_arr = cuda.to_device(cpu_arr)

# Use CuPy's asarray function and toDlpack to create DLPack capsule. There are multiple other ways to do this (i.e. PyTorch Utils)
dlpack_arr = cp.asarray(numba_gpu_arr).toDlpack()

# Migrate from Numba, used for custom CUDA JIT kernels to PyTorch
tf_tensor = tfdlpack.from_dlpack(dlpack_arr)

# Confirm TF tensor is on GPU
print(tf_tensor.device)

# Use `tfdlpack` to migrate back to Numba
dlpack_capsule = tfdlpack.to_dlpack(tf_tensor)
numba_arr = cuda.to_device(cp.fromDlpack(dlpack_capsule))
```

### PyTorch <-> TensorFlow
```python
# pip install torch
import torch
import tfdlpack
from torch.utils import dlpack as th_dlpack

# Torch - GPU Array
gpu_arr = torch.rand(10_000, 10_000).cuda()

# Use Torch's DLPack function to get DLPack Capsule
dlpack_arr = th_dlpack.to_dlpack(gpu_arr)

# Use `tfdlpack` to migrate to TensorFlow
tf_tensor = tfdlpack.from_dlpack(dlpack_arr)

# Confirm TF tensor is on GPU
print(tf_tensor.device)

# Use `tfdlpack` to migrate back to PyTorch
dlpack_capsule = tfdlpack.to_dlpack(tf_tensor)
torch_arr = th_dlpack.from_dlpack(dlpack_capsule)
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
