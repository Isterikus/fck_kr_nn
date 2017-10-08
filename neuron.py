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
				all_x[u_id] = {}
			all_x[u_id][int(line[1])] = [[float(s) if s != '' else 0.0 for s in line[2:]]]
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
		if elem in y:
			x[elem] = all_x[elem]

def get_data_y():
	with open('data/train.txt') as f:
		for line in f:
			line = line.strip().split('\t')
			rez = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
			y[int(line[0])] = rez[int(line[1])]

def model(inp):
	hidden_1_layer = {'V': {'weights': tf.Variable(tf.random_normal(42)), 'bias': tf.Variable(tf.random_normal(1))},
					'T': {'weights': tf.Variable(tf.random_normal(4)), 'bias': tf.Variable(tf.random_normal(1))},
					'M': {'weights': tf.Variable(tf.random_normal(1)), 'bias': tf.Variable(tf.random_normal(1))}}
	output_layer = {'weights': tf.Variable(tf.random_normal(3)), 'bias': tf.Variable(tf.random_normal(1))}
	l1 = {}
	for bem in inp:
		l1[bem] = tf.add(tf.matmul(inp[bem],hidden_1_layer[bem]['weights']), hidden_1_layer[bem]['bias'])
		lo = tf.matmul(l1[bem], output_layer['weights'])
	lo = tf.add(lo, output_layer['bias'])
	return lo

get_data_y()
get_data_x()

# np.random.seed(0)

tf_data_x = tf.placeholder(tf.float32, shape=[None,42,4,1]) # узел на который будем подавать аргументы функции
tf_data_y = tf.placeholder(tf.float32, shape=[None,4]) # узел на который будем подавать значения функции

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












