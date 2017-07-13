#!/usr/bin/env python3
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create variable with scalar

a = tf.Variable(2, name = "scalar")

# create variable with vector

b = tf.Variable([2,2], name = "vector")

# create variable as matrix

c = tf.Variable([[2,2], [3,2]], name = "matrix")

# create variable W with a 728 x 10 tensor, filled with zero's

W = tf.Variable(tf.zeros([728,10])) 

#only initiating but not fetching
#init = tf.global_variables_initializer()

init_ab = tf.variables_initializer([a, b], name = "init_ab")

with tf.Session() as sess:
	sess.run(init_ab)
	print(W.eval())