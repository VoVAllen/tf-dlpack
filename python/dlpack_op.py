import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
# from .capsule_api import to_capsule
import capsule_api

dlpack_ops = load_library.load_op_library(
    '/home/ubuntu/dev/tfdlpack/build/libtfdlpack.so')
to_dlpack = dlpack_ops.to_dlpack

a = tf.constant([10.1, 20.0, 11.2, -30.3])
b = to_dlpack(a)
capsule = capsule_api.to_capsule(b)
import  torch.utils.dlpack
th_tensor = torch.utils.dlpack.from_dlpack(capsule)
print(th_tensor)
print(th_tensor.device)