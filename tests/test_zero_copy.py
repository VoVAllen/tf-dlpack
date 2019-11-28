import tfdlpack
import torch as th
from torch.utils.dlpack import from_dlpack, to_dlpack
from tfdlpack import from_dlpack as tf_from_dlpack
import pytest

def get_gpu_memory_used():
    import gpustat
    gpu_query = gpustat.GPUStatCollection.new_query()
    gmem_used = gpu_query[0].memory_used
    return gmem_used


def get_capsule():
    a = th.ones([10000, 10000]).float().cuda()
    dl_cap = to_dlpack(a)
    return dl_cap

@pytest.mark.skip
def test_zero_copy():
    m1 = get_gpu_memory_used()
    c = get_capsule()
    m2 = get_gpu_memory_used()
    tf_t = tf_from_dlpack(c)
    m3 = get_gpu_memory_used()
    print(m1)
    print(m2)
    print(m3)

if __name__ == "__main__":
    test_zero_copy()
