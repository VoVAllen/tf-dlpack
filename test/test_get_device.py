import tfdlpack

import torch as th
from torch.utils.dlpack import from_dlpack, to_dlpack

a = th.tensor([1, 2, 3]).float()

dl_cap = to_dlpack(a)

device = tfdlpack.get_device(dl_cap)
print(device)