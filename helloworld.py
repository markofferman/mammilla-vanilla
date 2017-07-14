#!/usr/bin/env python3
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable(10, name = 'x')
y = tf.Variable(20, name = 'y')
z = tf.add(x, y)

# the proper way:
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	sess.run(tf.global_variables_initializer())
	for _ in range(10):
		sess.run(z)
		writer.close()

# the lazy loading way (run z-op in the loop):
#with tf.Session() as sess:
#	writer = tf.summary.FileWriter('./graphs', sess.graph)
#	sess.run(tf.global_variables_initializer())
#	for _ in range(10):
#		sess.run(tf.add(x,y))
#		writer.close()