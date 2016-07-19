from functools import reduce

import tensorflow as tf
import numpy as np
import transforms

s=tf.Session()

a=tf.ones([2,3])
b=tf.ones([2,1])+tf.constant(np.array([1,4]),shape=[2,1],dtype=tf.float32)
print(s.run(a/b))

c=tf.fill(tf.pack([tf.shape(a)[0]]), tf.squeeze(1))
print(s.run(tf.shape(c)))