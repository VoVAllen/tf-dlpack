import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
import ctypes
from .capsule_api import to_capsule, get_capsule_address


def get_op_files():
    import os
    pwd = os.getcwd()
    dir1 = os.path.abspath(os.path.join(
        os.getcwd(), os.pardir, "libtfdlpack.so"))
    dir2 = os.path.abspath(os.path.join(
        os.getcwd(), os.pardir, os.pardir, "build", "libtfdlpack.so"))
    if os.path.exists(dir1):
        return dir1
    if os.path.exists(dir2):
        return dir2
    return "libtfdlpack.so"


dlpack_ops = load_library.load_op_library(
    get_op_files())
_to_dlpack_add = dlpack_ops.to_dlpack
_from_dlpack = dlpack_ops.from_dlpack
_get_device_and_dtype = dlpack_ops.get_device_and_dtype


def to_dlpack(tf_tensor):
    return to_capsule(_to_dlpack_add(tf_tensor))


def get_device_and_dtype(dl_capsule):
    ptr = get_capsule_address(dl_capsule)
    with tf.device('/cpu:0'):
        ad_tensor = tf.constant([ptr], dtype=tf.uint64)
        return _get_device_and_dtype(ad_tensor).numpy()


def from_dlpack(dl_capsule):
    device_and_dtype = get_device_and_dtype(dl_capsule)
    device = device_and_dtype[:2]
    dtype = device_and_dtype[2]
    # print("Dtype: {}".format(dtype))
    ptr = get_capsule_address(dl_capsule, consume=True)
    # tf_device_type =
    if device[0] == 1:
        tf_device_type = "cpu"
        tf_device_id = int(device[1])
    elif device[0] == 2:
        tf_device_type = "gpu"
        tf_device_id = int(device[1])
    else:
        raise RuntimeError("Unsupported Device")
    tf_device = "/{}:{}".format(tf_device_type, tf_device_id)
    with tf.device(tf_device):
        ad_tensor = tf.constant([ptr], dtype=tf.uint64)
        # tf_tensor = _from_dlpack(ad_tensor, T=tf.float32)
        tf_tensor = _from_dlpack(ad_tensor, T=tf.DType(dtype))

    return tf_tensor
