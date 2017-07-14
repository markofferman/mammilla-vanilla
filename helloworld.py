#!/usr/bin/env python3
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create operations
a = tf.add(2, 5)
b = tf.multiply(a, 3)

# start up a session
sess = tf.Session()

# replace dictionairy with something else
replace_dict = {a: 15}

# run the session with the replacement

print(sess.run(b, feed_dict = replace_dict))