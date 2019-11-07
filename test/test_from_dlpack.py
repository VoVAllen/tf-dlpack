import tfdlpack

from tfdlpack import from_dlpack as tf_from_dlpack

from torch.utils.dlpack import from_dlpack, to_dlpack
import torch as th

def get_capsule():
    a = th.tensor([1, 2, 3]).float()
    dl_cap = to_dlpack(a)
    return dl_cap

tf_t = tf_from_dlpack(get_capsule())
print(tf_t)

