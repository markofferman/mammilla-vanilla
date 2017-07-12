import tensorflow as tf

a = tf.constant(3)
b = tf.constant(2)
x = tf.add(a, b)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(x))

#Close it when done
writer.close()


#run python helloworld.py
#run tensorboard --logdir="./graphs"