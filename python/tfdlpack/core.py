# pylint: disable=no-name-in-module, invalid-name
"""core functions"""

import tensorflow as tf
from tensorflow.python.framework import load_library

from . import libinfo
from .capsule_api import to_capsule, get_capsule_address

# version number
__version__ = libinfo.__version__

dlpack_ops = load_library.load_op_library(libinfo.find_lib_path()[0])
_to_dlpack_address = dlpack_ops.to_dlpack
_from_dlpack = dlpack_ops.from_dlpack
_get_device_and_dtype = dlpack_ops.get_device_and_dtype


def to_dlpack(tf_tensor):
    """Convert the given tensorflow tensor to DLPack format.
    """
    with tf.device(tf_tensor.device):
        cap = to_capsule(_to_dlpack_address(tf_tensor))
    return cap


def get_device_and_dtype(dl_capsule):
    """Get capsule's device and its corresponding dtype
    First element is device type and second is device id (according to DLPack protocal)
    Third element is the tf data type (can be convert to tf type by tf.DType(d) )
    """
    ptr = get_capsule_address(dl_capsule)
    with tf.device('/cpu:0'):
        ad_tensor = tf.constant([ptr], dtype=tf.uint64)
        return _get_device_and_dtype(ad_tensor).numpy()


def from_dlpack(dl_capsule):
    """Convert capsule to tf tensor"""
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
    with tf.device("cpu:0"):
        ad_tensor = tf.constant([ptr], dtype=tf.uint64)
    with tf.device(tf_device):
        # tf_tensor = _from_dlpack(ad_tensor, T=tf.float32)
        tf_tensor = _from_dlpack(ad_tensor, T=tf.DType(dtype))

    return tf_tensor
