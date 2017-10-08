import csv
import pandas as pd

rez = {}
right = 0
false = 0
with open('try.csv') as f:
	with open('data/temp_new.csv') as f2:
		rez = csv.reader(f)
		rez = [i for i in rez]
		val = csv.reader(f2, delimiter='\t')
		val = [i for i in val]
		for i in range(len(rez)):
			if int(rez[i][1]) == int(val[i][-1]):
				right += 1
			else:
				false += 1
		print('r=', right, 'f=', false)


# rez = pd.read_csv('try.csv')

# val = pd.read_csv('data/temp_new.csv', delimiter='\t')

# i = 1
# while i <= 6299:
# 	row = next(rez.iterrows())[i]
# 	print(row)
# 	i += 1