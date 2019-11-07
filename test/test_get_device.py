import tfdlpack
import tensorflow as tf
import torch as th
import numpy as np
from torch.utils.dlpack import from_dlpack, to_dlpack

types = [np.float16, np.float32, np.float32,
         np.int8, np.int16, np.int32, np.int64]
devices = {
    1: lambda t: t.cpu(),
    2: lambda t: t.cuda(0),
}


def test_get_op():
    for np_type in types:
        for kDLContext, th_device in devices.items():
            np_array = np.array([1, 2, 3], dtype=np_type)
            th_tensor = th_device(th.tensor(np_array))
            dl_cap = to_dlpack(th_tensor)
            tf_device_and_dtype = tfdlpack.get_device_and_dtype(dl_cap)
            device_id = th_tensor.device.index
            device_id = 0 if device_id is None else device_id
            assert kDLContext == tf_device_and_dtype[0].item()
            assert device_id == tf_device_and_dtype[1].item()
            assert tf.DType(
                tf_device_and_dtype[2].item()).as_numpy_dtype == np_type


if __name__ == "__main__":
    test_get_op()
