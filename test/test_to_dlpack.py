import tfdlpack
import tensorflow as tf
import torch.utils.dlpack

def get_capsule():
    a = tf.ones([1000, 100000])
    capsule = tfdlpack.to_dlpack(a)
    return capsule


capsule = get_capsule()
print(capsule)

# Apply for same 
a = tf.zeros([1000, 100000])

# Torch won't do copy at this time
th_tensor = torch.utils.dlpack.from_dlpack(capsule)

# But when you need to print it or do computation with it, torch will also copy the memory
# Checked with cupy's dlpack, also same behaviors
print(th_tensor)
print(th_tensor.device)

