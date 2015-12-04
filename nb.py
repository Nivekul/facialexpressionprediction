import numpy as np
from sklearn.naive_bayes import *

def GaussianNaiveBayes(inputs_train, inputs_valid, target_train, target_valid, partialFit = False):
	name = "Gaussian Naive Bayes"
	clf = GaussianNB()
	if partialFit:
		name += "(Partial Fit)"
		clf.partial_fit(inputs_train, np.ravel(target_train), [1,2,3,4,5,6,7])
	else:
		clf.fit(inputs_train, np.ravel(target_train))
	prediction = clf.predict(inputs_valid)
	# print prediction
	correct = np.count_nonzero(np.ravel(target_valid) == prediction)
	total = target_valid.shape[0]
	correctRate = (float(correct)/total)*100

	return name, correctRate

def MultinomialNaiveBayes(inputs_train, inputs_valid, target_train, target_valid, partialFit = False):
	name = "Multinomial Naive Bayes"
	clf = MultinomialNB()
	if partialFit:
		name += "(Partial Fit)"
		clf.partial_fit(inputs_train, np.ravel(target_train), [1,2,3,4,5,6,7])
	else:
		clf.fit(inputs_train, np.ravel(target_train))
	prediction = clf.predict(inputs_valid)
	correct = np.count_nonzero(np.ravel(target_valid) == prediction)
	total = target_valid.shape[0]
	correctRate = (float(correct)/total)*100

	return name, correctRate

def BernoulliNaiveBayes(inputs_train, inputs_valid, target_train, target_valid, partialFit = False):
	name = "Bernoulli Naive Bayes"
	clf = BernoulliNB()
	if partialFit:
		name += "(Partial Fit)"
		clf.partial_fit(inputs_train, np.ravel(target_train), [1,2,3,4,5,6,7])
	else:
		clf.fit(inputs_train, np.ravel(target_train))
	prediction = clf.predict(inputs_valid)
	correct = np.count_nonzero(np.ravel(target_valid) == prediction)
	total = target_valid.shape[0]
	correctRate = (float(correct)/total)*100

	return name, correctRate

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	print "-"*71
	GaussianNaiveBayes()
	GaussianNaiveBayes(True)
	MultinomialNaiveBayes()
	MultinomialNaiveBayes(True)
	BernoulliNaiveBayes()
	BernoulliNaiveBayes(True)
