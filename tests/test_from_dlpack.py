import tfdlpack
import numpy as np
from tfdlpack import from_dlpack as tf_from_dlpack
import tensorflow as tf

from torch.utils.dlpack import from_dlpack as th_from_dlpack
from torch.utils.dlpack import to_dlpack as th_to_dlpack
import torch as th
th.cuda.init()

types = [np.float16, np.float32, np.float32,
         np.int8, np.int16, np.int32, np.int64]
devices = {
    "cpu": lambda t: t.cpu(),
    "gpu": lambda t: t.cuda(0),
}


def get_context_from_tf(device_str):
    dspec = tf.DeviceSpec.from_string(device_str)


def test_from_torch_to_tf():
    for np_type in types:
        for th_device_name, th_device in devices.items():
            np_array = np.array([1, 2, 3], dtype=np_type)
            th_tensor = th_device(th.tensor(np_array))
            dl_cap = th_to_dlpack(th_tensor)
            tf_tensor = tf_from_dlpack(dl_cap)
            th_device_id = th_tensor.device.index
            th_device_id = 0 if th_device_id is None else th_device_id
            tf_device = tf.DeviceSpec.from_string(tf_tensor.device)
            assert th_device_name == tf_device.device_type.lower()
            assert th_device_id == tf_device.device_index
            assert np.array_equal(th_tensor.cpu().numpy(), tf_tensor.numpy())


if __name__ == "__main__":
    test_from_torch_to_tf()
