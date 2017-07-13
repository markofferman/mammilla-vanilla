import tensorflow as tf 

my_const = tf.constant([1.0,2.0], name = "my_const")
print(tf.get_default_graph().as_graph_def())