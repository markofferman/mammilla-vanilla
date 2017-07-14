#!/usr/bin/env python3
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

W = tf.Variable(10)
assign_op = W.assign(100)

with tf.Session() as sess:
	sess.run(assign_op)
	print(W.eval())