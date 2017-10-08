from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
import numpy as np
# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
# print(mnist.train.next_batch(6))
# print('HO----------------------------------------------------------------------------')
# exit()
# Parameters
num_epoch = 4
num_steps = 100 # Total steps to train
batch_size = 6229 # The number of samples per batch
num_classes = 4 # The 4 digits
num_features = 46 # Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

x = []
y = []

def get_data():
	with open('data/res.txt') as f:
		for line in f:
			line = line.split('\t')
			line[-1] = line[-1][:-1]
			x.append([float(i) if i != 'NA' else 0.0 for i in line[2:-1]])
			y.append(int(line[-1]))

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op = forest_graph.inference_graph(X)
print(infer_op)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

get_data()
train_x = x[:4500]
train_y = y[:4500]
test_x = x[4500:]
test_y = y[4500:]

# Training
for i in range(num_epoch):
		# Prepare Data
		# Get the next batch of MNIST data (only images are needed, not labels)
		# batch_x, batch_y = mnist.train.next_batch(batch_size)
		for j in range(100):
			fr = 45 * j
			to = 45 * (j + 1)
			batch_x = train_x[fr:to]
			batch_y = train_y[fr:to]
			_, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
			if j % 10 == 0:
				acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
				print('Epoch %i, Step %i, Loss: %f, Acc: %f' % (i, j, l, acc))

# Test Model
# test_x, test_y = test_x, test_y
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))