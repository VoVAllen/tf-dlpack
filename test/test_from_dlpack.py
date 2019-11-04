import tfdlpack

from tfdlpack import from_dlpack as tf_from_dlpack

# tf_from_dlpack = dlpackop.from_dlpack

from torch.utils.dlpack import from_dlpack, to_dlpack
import torch as th

a = th.tensor([1, 2, 3]).float()

dl_cap = to_dlpack(a)

tf_t = tf_from_dlpack(dl_cap)
print(tf_t)