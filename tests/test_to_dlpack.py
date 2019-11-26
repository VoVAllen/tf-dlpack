import tfdlpack
import numpy as np
from tfdlpack import from_dlpack as tf_from_dlpack
from tfdlpack import to_dlpack as tf_to_dlpack

from torch.utils.dlpack import from_dlpack as th_from_dlpack
from torch.utils.dlpack import to_dlpack as th_to_dlpack
import tensorflow as tf
import torch as th

# tf.config.experimental.set_device_policy("explicit") # This will raise error when type is tf.int32

types = [np.float16, np.float32, np.float32,
         np.int8, np.int16, np.int32, np.int64]

devices = {
    ("cpu", 0): tf.device("/cpu:0"),
    ("cuda", 0):  tf.device("/gpu:0"),
}


def get_context_from_tf(device_str):
    dspec = tf.DeviceSpec.from_string(device_str)


def test_from_tf_to_torch():
    for np_type in types:
        for tf_device_name, tf_device in devices.items():
            np_array = np.array([1, 2, 3], dtype=np_type)
            with tf_device:
                tf_tensor = tf.constant(np_array) # TF Bug: when np_type is int32, the tf_tensor would be placed on CPU when with GPU device
                tf_tensor = tf.identity(tf_tensor) # See: https://github.com/tensorflow/tensorflow/issues/34071
            dl_cap = tf_to_dlpack(tf_tensor)
            th_tensor = th_from_dlpack(dl_cap)
            th_device_id = th_tensor.device.index
            th_device_id = 0 if th_device_id is None else th_device_id
            th_device_name = th_tensor.device.type
            assert th_device_name.lower() == tf_device_name[0].lower()
            assert th_device_id == tf_device_name[1]
            assert np.array_equal(th_tensor.cpu().numpy(), tf_tensor.numpy())


if __name__ == "__main__":
    test_from_tf_to_torch()
