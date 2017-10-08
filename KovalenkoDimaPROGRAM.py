from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize,normalize
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np

# For printing in Jupyter
def printall(X,max_rows=46):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))

# Prints plot(barh) in order to know the importance of the input parameters
def analyze_vars(X, y):
	model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
	model.fit(X, y)
	roc = roc_auc_score(y, model.oob_prediction_)
	feature_importances = pd.Series(model.feature_importances_, index=X.columns)
	feature_importances.sort_values(inplace=True)
	feature_importances.plot(kind="barh", figsize=(7,12))
	pl.show()

# Analyzing n_jobs in order to increase the speed
def analyze_n_jobs(X, y):
	print("Analyzing n_jobs.")
	model = RandomForestRegressor(100, oob_score=True, n_jobs=1, random_state=42)
	model.fit(X, y)

	model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
	model.fit(X, y)
	'''
	12.4 s ± 11.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
	9.43 s ± 212 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
	'''

# Analyzing estimator in order to increase the score
def analyze_estimator(X, y):
	print("Analyzing estimator.")
	res = []
	n_estimator_opts = [30,50,100,200,500,1000,2000]

	for trees in n_estimator_opts:
		model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
		model.fit(X, y)
		print('trees = ', trees)
		roc = roc_auc_score(y, model.oob_prediction_)
		print('score: ', roc)
		res.append(roc)

	pd.Series(res, n_estimator_opts).plot()
	pl.show()
	'''
	trees =  30
	c-stat:  0.572312062065
	trees =  50
	c-stat:  0.583948421134
	trees =  100
	c-stat:  0.599601643246
	trees =  200
	c-stat:  0.607823321892
	trees =  500
	c-stat:  0.612932251593
	trees =  1000
	c-stat:  0.617246186174
	trees =  2000
	c-stat:  0.620618505885
	'''

# Analyzing max_feature in order to increase the score
def analyze_max_feature(X, y):
	print("Analyzing max_feature")
	max_featured_opts = ['auto','sqrt','log2',0.9,0.2]

	for max_feature in max_featured_opts:
		model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_feature)
		model.fit(X, y)
		print('opt = ', max_feature)
		roc = roc_auc_score(y, model.oob_prediction_)
		print('score = ', roc)
	'''
	opt =  auto
	c-stat =  0.616470311033
	opt =  sqrt
	c-stat =  0.620629055348
	opt =  log2
	c-stat =  0.62343741753
	opt =  0.9
	c-stat =  0.617969662274
	opt =  0.2
	c-stat =  0.624980735669
	'''

def analyze_min_samples_leaf(X, y):
	print("Analyzing min_samples_leaf")
	res = []
	min_samples_leaf_opts = [(i * 5) for i in range(1,11)]

	for min_sample in min_samples_leaf_opts:
		model = RandomForestRegressor(n_estimators=1000,
										oob_score=True,
										n_jobs=-1,
										random_state=42,
										max_features=0.2,
										min_samples_leaf=min_sample)
		model.fit(X, y)
		print('sample = ', min_sample)
		roc = roc_auc_score(y, model.oob_prediction_)
		print('score = ', roc)
		res.append(roc)

	pd.Series(res, min_samples_leaf_opts).plot()
	pl.show()
	'''
	sample =  5
	c-stat =  0.629317765215
	sample =  10
	c-stat =  0.629739442888
	sample =  15
	c-stat =  0.630293431072
	sample =  20
	c-stat =  0.632172710667
	sample =  25
	c-stat =  0.630655721536
	sample =  30
	c-stat =  0.629575349083
	sample =  35
	c-stat =  0.631335033393
	sample =  40
	c-stat =  0.630186354152
	sample =  45
	c-stat =  0.63043266061
	sample =  50
	c-stat =  0.629611539296
	'''

all_x = [[] for i in range(10000)]
x = [[] for i in range(10000)]
test_y = pd.read_csv('test.txt', delimiter='\t')
test_y = test_y.drop(['TARGET'], axis=1)
def get_data_x():
	with open('data/Base1.txt') as f:
		i = 1
		for line in f:
			line = line.split('\t')
			line[-1] = line[-1][:-1]
			u_id = int(line[0]) - 1
			if not u_id in all_x:
				# all_x[u_id] = {}
				all_x[u_id] = []
			all_x[u_id].append([float(s) if s != '' else 0.0 for s in line[2:]])
			i += 1
	for k in range(len(all_x)):
		part = all_x[k]
		ln = len(part)
		sr = [0.0 for i in range(42)]
		j = 0
		for dt in part:
			j += 1
			sr = np.add(sr, dt)
		sr = np.divide(sr, j)
		all_x[k] = [i for i in sr]
	with open('Base2.txt') as f:
		for line in f:
			line = line.split('\t')
			line[-1] = line[-1][:-1]
			elems = [i for i in line[1:]]
			for srav in elems:
				all_x[int(line[0])].append(int(srav,16))
	for i in range(len(all_x)):
		if (i + 1) in test_y['ID']:
			x[i] = all_x[i]

def write_test(X, y):
	test_x = pd.read_csv('Base1.txt', delimiter='\t')
	t = pd.read_csv('Base2.txt', delimiter='\t')
	get_data_x()
	with open('test.txt') as f:
		model = RandomForestRegressor(n_estimators=1000,
										oob_score=True,
										n_jobs=-1,
										random_state=42,
										max_features=0.2,
										min_samples_leaf=20)
		model.fit(X, y)
		i = 0
		# for srav in x:
		# 	if len(srav) == 0:
		# 		print(i)
		# 		del x[i]
		# 	i += 1
		for i in x:
			if len(i) != 47:
				print(i)
				exit()
		test_y.insert(1,'TARGET', model.predict([test for test in x[:3037]]))
		test_y.to_csv('KovalenkoDima_test.txt', delimiter='\t', index=False)

if __name__ == "__main__":

	# Reading data parsed into 1 file [ID,V1...V42,T1...T4,TARGET]
	# Was not sure what to do with T1-T4 so just add it to input...
	X = pd.read_csv('new.tsv', delimiter='\t')
	y = X.pop("TARGET")
	y = label_binarize(y, classes=[0, 1, 2, 3])

	# Fill missing vars
	for col in X:
		X[col].fillna(X.eval(col).mean(), inplace=True)

	# Remove the non-affecting columns
	# analyze_vars(X, y)
	for col in ['V18','V11','V2','V9','V13','V14']:
		X = X.drop([col], axis=1)
	write_test(X,y)
	exit()
	analyze_n_jobs(X, y)
	analyze_estimator(X, y)
	analyze_max_feature(X, y)
	analyze_min_samples_leaf(X, y)

	# Final model:
	model = RandomForestRegressor(n_estimators=1000,
									oob_score=True,
									n_jobs=-1,
									random_state=42,
									max_features=0.2,
									min_samples_leaf=20)
	model.fit(X, y)
	roc = roc_auc_score(y, model.oob_prediction_)
	print('Final score = ', roc)
	# Final score = 0.632768439621
	write_test(X,y)
