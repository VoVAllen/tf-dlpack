import tfdlpack
import tensorflow as tf
import torch.utils.dlpack
a = tf.constant([10.1, 20.0, 11.2, -30.3])
capsule = tfdlpack.to_dlpack(a)
# capsule = capsule_api.to_capsule(b)
th_tensor = torch.utils.dlpack.from_dlpack(capsule)
print(th_tensor)
print(th_tensor.device)

th_tensor[1] = 999
print(th_tensor)
# from tensorflow import conte
print(a.numpy())
