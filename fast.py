import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import csv
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

samples = 24 # количество точек
packetSize = 4 # размер пакета
sigma = 0.01 # среднеквадратическое отклонение шум
hm_epochs = 300

samples = 6229

all_x = {}
x = {}
y = {}

def get_data_x():
	with open('data/Base1.txt') as f:
		i = 1
		for line in f:
			line = line.split('\t')
			line[-1] = line[-1][:-1]
			u_id = int(line[0])
			if not u_id in all_x:
				# all_x[u_id] = {}
				all_x[u_id] = []
			all_x[u_id].append([float(s) if s != '' else 0.0 for s in line[2:]])
			i += 1
		# while i < 6229:
		# 	j = 0
		# 	line = f.readline().strip().split('	')
		# 	u_id = int(line[0]) - 1
		# 	x[u_id] = [[] for k in range(6)]
		# 	x[u_id][int(line[1]) - 1] = [float(k) if k != '' else 0.0 for k in line[2:]]
		# 	while j < 5:
		# 		line = f.readline().strip().split(' ')
		# 		x[u_id][int(line[1]) - 1] = [float(k) if k != '' else 0.0 for k in line[2:]]
		# 		j += 1
		# 	i += 1
	for elem in all_x:
		part = all_x[elem]
		ln = len(part)
		sr = [0.0 for i in range(42)]
		j = 0
		for dt in part:
			j += 1
			sr = np.add(sr, dt)
		sr = np.divide(sr, j)
		all_x[elem] = [i for i in sr]
	with open('data/Base2.txt') as f:
		for line in f:
			line = line.split('\t')
			line[-1] = line[-1][:-1]
			elems = [int(i,base=16) for i in line[1:]]
			for srav in elems:
				all_x[int(line[0])].append(srav)
	for elem in all_x:
		if elem in y:
			x[elem] = all_x[elem]

def get_data_y():
	with open('data/train.txt') as f:
		for line in f:
			line = line.strip().split('\t')
			rez = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
			y[int(line[0])] = rez[int(line[1])]

# def model(x):
# 	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([47,47])), 'biases': tf.Variable(47)}
# 	output_layer = {'weights': tf.Variable(tf.random_normal([47,4])), 'biases': tf.Variable(tf.random_normal(4))}
	
# 	l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']) + hidden_1_layer['biases'])
# 	l1 = tf.nn.relu(l1)
# 	lo = tf.matmul(l1,output_layer['weights']) + output_layer['biases']
# 	return lo

# def train():
# 	train_x = {}
# 	train_y = {}
# 	test_x = {}
# 	test_y = {}
# 	for i in x:
# 		if i < 4500:
# 			train_x[i] = x[i]
# 			train_y[i] = y[i]
# 		else:
# 			test_x[i] = x[i]
# 			test_y[i] = y[i]
# 	prediction = model(x)
# 	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
# 	optimizer = tf.train.AdamOptimizer().minimize(cost)

# 	hm_epochs = 300

# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())

# 		for epoch in range(hm_epochs):
# 			epoch_loss = 0
# 			for _ in range(len(x)):
# 				_, c = sess.run([optimizer,cost], feed_dict={x: train_x, y: train_y})
# 				epoch_loss += c
# 			print('Epoch ', epoch, ' | loss ', epoch_loss)

# 		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 		acc = tf.reduce_mean(tf.cast(correct,'float'))
# 		print('Acc ', acc.eval({x: test_x, y: test_y}))

get_data_y()
get_data_x()
print(len(x), len(y))
# np.random.seed(0)

tf_data_x = tf.placeholder(tf.float32, shape=(46,)) # узел на который будем подавать аргументы функции
tf_data_y = tf.placeholder(tf.float32, shape=(4,)) # узел на который будем подавать значения функции

weight = tf.Variable(initial_value=0.0, dtype=tf.float32)
bias = tf.Variable(initial_value=0.0, dtype=tf.float32)
model = tf.add(tf.multiply(tf_data_x, weight), bias)

loss = tf.reduce_mean(tf.square(model-tf_data_y)) # функция потерь, о ней ниже
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss) # метод оптимизации, о нём тоже ниже

with tf.Session() as session:
	tf.global_variables_initializer().run()
	for epoch in hm_epochs:
		for i in x:
			feed_dict = {tf_data_x: x[i], tf_data_y: y[i]}
			# print(feed_dict)
			_, l = session.run([optimizer, loss], feed_dict=feed_dict) # запускаем оптимизатор и вычисляем "потери"
			print("ошибка: %f" % (l, ))
			print("a = %f, b = %f" % (weight.eval(), bias.eval()))












