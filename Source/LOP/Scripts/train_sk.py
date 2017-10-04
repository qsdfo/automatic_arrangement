#!/usr/bin/env python
# -*- coding: utf8 -*-


def train_sk(model,
		  	 piano_train, orch_train, train_index,
		  	 piano_valid, orch_valid, valid_index,
		  	 parameters, config_folder, start_time_train, logger_train):

	

	# train
	classifier.fit(X_train, y_train)

	# predict
	predictions = classifier.predict(X_test)

	return predictions


if __name__ == '__main__':
	from sklearn.datasets import make_multilabel_classification
	from sklearn.model_selection import train_test_split
	from skmultilearn.problem_transform import LabelPowerset
	from sklearn.naive_bayes import GaussianNB


	# this will generate a
	X, y = make_multilabel_classification(sparse=True, n_labels = 5,
  		return_indicator='sparse', allow_unlabeled = False)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

	classifier = LabelPowerset(GaussianNB())

	import pdb; pdb.set_trace()

	# train
	classifier.fit(X_train, y_train)

	# predict
	predictions = classifier.predict(X_test)