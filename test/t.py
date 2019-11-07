import tensorflow as tf

with tf.device("/gpu:0"):
    # a=tf.constant([111], dtype=tf.float32)
    a=tf.constant([111])
    # a=tf.identity(a)
print(a.device)
print(a.dtype)