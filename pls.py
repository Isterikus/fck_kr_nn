from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

X = pd.read_csv('res2.csv', delimiter='\t')
y = X.pop("TARGET")
y = label_binarize(y, classes=[0, 1, 2, 3])
'''
'''
X.describe()
'''
'''
for bem in X:
	X[bem].fillna(X.eval(bem).mean(), inplace=True)
# hz nado li

numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head()

# end hz nado li

# 100 True 42
model = RandomForestRegressor(n_estimators=200,oob_score=True,random_state=42)

model.fit(X[numeric_variables], y)

print(model.oob_score_)

y_oob = model.oob_prediction_
print("c-stat: ", roc_auc_score(y, y_oob))


for i in range(1,5):
	var = 'T' + str(i)
	dummies = pd.get_dummies(X[var], prefix=var)
	X = pd.concat([X, dummies], axis=1)
	X.drop([var], axis=1, inplace=True)


model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)
prin("c-stat: ", roc_auc_score(y, model.oob_prediction_))


model.feature_importances_

'''

# !!! ROC !!!
ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(X, y, test_size=0.25) 

model_rfc = RandomForestClassifier(n_estimators = 70) #в параметре передаем кол-во деревьев
model_knc = KNeighborsClassifier(n_neighbors = 18) #в параметре передаем кол-во соседей
model_lr = LogisticRegression(penalty='l1', tol=0.01) 
model_svc = svm.SVC() #по умолчанию kernek='rbf'

scores = cross_validation.cross_val_score(model_rfc, X, y, cv = kfold)
itog_val['RandomForestClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_knc, X, y, cv = kfold)
itog_val['KNeighborsClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(model_lr, X, y, cv = kfold)
itog_val['LogisticRegression'] = scores.mean()
scores = cross_validation.cross_val_score(model_svc, X, y, cv = kfold)
itog_val['SVC'] = scores.mean()

DataFrame.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False)


'''