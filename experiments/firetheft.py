#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import xlrd

import utils


DATA_FILE = './data/fire_theft.xls'

# Step 1. Read in the data
book = xlrd.open_workbook(DATA_FILE, encoding_override = "utf-8")
sheet = book.sheet_by_index(0)
data = np.array([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2. Create placeholders for input X and label Y
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

# Step 3. Create Variable weight and bias
w = tf.Variable(0.0, name = "weights_1")
b = tf.Variable(0.0, name = "bias")

# Step 4. Construct model to predict Y
Y_predicted = X * w + b

# Step 5. Construct a loss function with the square error
## loss = tf.square(Y - Y_predicted, name = "loss")

# Extra. Huber loss
def huber_loss(labels, predictions, delta = 1.0):
	residual = tf.abs(predictions - labels)
	condition = tf.less(residual, delta)
	small_res = 0.5 * tf.square(residual)
	large_res = delta * residual - 0.5 * tf.square(delta)
	return tf.where(condition, small_res, large_res)


# Step 6. Construct optimizer with gradient decient with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(huber_loss(Y, Y_predicted))

with tf.Session() as sess:
	# Step 7. Initialize the global variables (w + b)
	sess.run(tf.global_variables_initializer())

	writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

	# Step 8. train the model with 100 epochs
	for i in range(50):
		total_loss = 0
		# with the data
		for x, y in data:
			# session runs train_op to minimize loss
			sess.run(optimizer, feed_dict={X: x, Y: y})
		print('epoch {0}: {1}'.format(i, total_loss/n_samples))

	writer.close()

	# Step 9. Output the values of w & b
	w, b = sess.run([w, b])
	print("Value of b is ", b)
	print("Value of w is", w)


# Step 10. Plot the results
# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()