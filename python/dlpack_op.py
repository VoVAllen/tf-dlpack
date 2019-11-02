import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
# from .capsule_api import to_capsule
import capsule_api

tvm_runtime_ops = load_library.load_op_library(
    '/home/ubuntu/dev/tfdlpack/build/libtfdlpack.so')
tvm_runtime = tvm_runtime_ops.tvm_runtime



a = tf.constant([10.1, 20.0, 11.2, -30.3])
b = tvm_runtime(a)
capsule = capsule_api.to_capsule(b)
import  torch.utils.dlpack
th_tensor = torch.utils.dlpack.from_dlpack(capsule)
print(th_tensor)
print(th_tensor.device)