import numpy, pandas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from math import sqrt

def load_abalone():
	# Import data
	data = numpy.genfromtxt('abalone_data.csv',delimiter=',',dtype=str)

	# Convert categorical data
	data = make_column_transformer((OneHotEncoder(), [0]), remainder='passthrough').fit_transform(data)

	# Shuffle
	numpy.random.seed(1567708903)
	shuffled_index = numpy.random.permutation(data.shape[0])
	data = data[shuffled_index]

	return data

def regTune(regType, regObj, regParam, train_X, train_Y, fold):
	# Hyperparameter tuning with 5-fold grid search cross validation
	gscv = GridSearchCV(regObj, regParam, scoring=make_scorer(r2_score),
		cv=5, iid=False, return_train_score=True).fit(train_X, train_Y)
	print("Best " + regType + " Params: " + str(gscv.best_params_))
	print("Best " + regType + " Mean Validation Score: " + str(gscv.best_score_))
	print("Total Tuning Time: "
		+ str(5*(sum(gscv.cv_results_['mean_fit_time']) + sum(gscv.cv_results_['mean_score_time']))) + " seconds")

	# Save table of necessary data
	df = pandas.DataFrame(gscv.cv_results_)
	df.to_csv(regType + "_" + str(fold) + ".csv")

	# Return best tuned estimator
	return gscv.best_estimator_

def regTest(regType, regObj, regParam, data):
	# Performance testing with 5-fold cross validation
	cv_scores = numpy.array([])

	fold = 5
	for i in range(fold):
		# Data Splice
		cutoff_s = int(data.shape[0]*float(i)/float(fold))
		cutoff_e = int(data.shape[0]*((float(i)/float(fold))+0.2))
		test_data = data[cutoff_s:cutoff_e]
		test_X = test_data[:, :-1].astype(float)
		test_Y = test_data[:, -1].reshape(-1, 1).astype(float)
		train_data = numpy.concatenate((data[:cutoff_s], data[cutoff_e:]), axis=0)
		train_X = train_data[:, :-1].astype(float)
		train_Y = train_data[:, -1].reshape(-1, 1).astype(float)

		# Tuning
		print("Fold " + str(i) + ":")
		regObj_best = regTune(regType, regObj, regParam, train_X, train_Y, i)

		# Testing
		cv_scores = numpy.append(cv_scores, regObj_best.score(test_X, test_Y))
		print(regType + " Testing Score: " + str(cv_scores[-1]) + "\n")

	# Confidence Interval
	print(regType + " Testing Mean: " + str(cv_scores.mean()))
	print(regType + " Testing Std Dev: " + str(cv_scores.std()))
	print(regType + " Testing Confidence Interval: ("
		+ str(cv_scores.mean()-(1.96*cv_scores.std()/sqrt(cv_scores.shape[0]))) + ", "
		+ str(cv_scores.mean()+(1.96*cv_scores.std()/sqrt(cv_scores.shape[0]))) + ")")

def main():
	## Load data
	data = load_abalone()

	## Choose regression type: lr, krr, knr, nn
	regType = "lr"

	if regType == "lr":
		# Data Splice
		cutoff = int(data.shape[0]*0.8)
		train_data = data[:cutoff]
		test_data = data[cutoff:]
		train_X = train_data[:, :-1].astype(float)
		train_Y = train_data[:, -1].reshape(-1, 1).astype(float)
		test_X = test_data[:, :-1].astype(float)
		test_Y = test_data[:, -1].reshape(-1, 1).astype(float)

		# Data Prepend
		train_Xp1 = numpy.concatenate((numpy.ones((train_X.shape[0], 1)), train_X), axis=1)
		test_Xp1 = numpy.concatenate((numpy.ones((test_X.shape[0], 1)), test_X), axis=1)

		# Linear Regression Testing
		lrObj = LinearRegression().fit(train_Xp1, train_Y)
		print(regType + " Score: " + str(lrObj.score(test_Xp1, test_Y)))

	elif regType == "krr":
		## Kernel Ridge Regression with a non-linear kernel
		krrObj = KernelRidge()
		krrParam = [{'kernel': ['polynomial'], 'degree': [2, 3, 4, 5],
					'alpha': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1],
					'gamma': [0.1, 0.3, 0.5, 0.7, 0.9, 1]},
					{'kernel': ['rbf'],
					'alpha': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1],
					'gamma': [0.1, 0.3, 0.5, 0.7, 0.9, 1]}]
		regTest(regType, krrObj, krrParam, data)

	elif regType == "knr":
		## k-Neighbors Regression
		knrObj = KNeighborsRegressor()
		knrParam = {'n_neighbors': [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'weights': ['uniform', 'distance'],
					'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']}
		regTest(regType, knrObj, knrParam, data)

	elif regType == "nn":
		## Neural Network with at least 2 hidden layers
		nnObj = MLPRegressor()
		nnParam = {'hidden_layer_sizes': [(2,), (3,)], 'activation': ['logistic', 'tanh'],
					'solver': ['sgd', 'lbfgs'], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
					'max_iter': [20000], 'learning_rate_init': [0.001, 0.01, 0.1]}
		regTest(regType, nnObj, nnParam, data)

	else:
		print("Regression type not available.")

if __name__ == '__main__':
	main()