from __future__ import print_function

import tensorflow as tf

a = tf.constant(3)
b = tf.constant(2)

with tf.Session() as sess:
	print("Addition with constants: %i" % sess.run(a+b))
