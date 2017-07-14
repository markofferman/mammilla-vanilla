#!/usr/bin/env python3
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

c = a + b

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./my_graph', sess.graph)
	print(sess.run(c, {a: [1, 2, 3]}))


writer.close()