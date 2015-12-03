import numpy as np
from sklearn.naive_bayes import *
from util import *

inputs_train, inputs_valid, target_train, target_valid = LoadData('labeled_images.mat', 0.3)

def GaussianNaiveBayes(partialFit = False):
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
	total, _ = target_valid.shape
	print name, ": ", float(correct)/total
	print "-"*71

def MultinomialNaiveBayes(partialFit = False):
	name = "Multinomial Naive Bayes"
	clf = MultinomialNB()
	if partialFit:
		name += "(Partial Fit)"
		clf.partial_fit(inputs_train, np.ravel(target_train), [1,2,3,4,5,6,7])
	else:
		clf.fit(inputs_train, np.ravel(target_train))
	prediction = clf.predict(inputs_valid)
	correct = np.count_nonzero(np.ravel(target_valid) == prediction)
	total, _ = target_valid.shape
	print name, ": ", float(correct)/total
	print "-"*71

def BernoulliNaiveBayes(partialFit = False):
	name = "Bernoulli Naive Bayes"
	clf = BernoulliNB()
	if partialFit:
		name += "(Partial Fit)"
		clf.partial_fit(inputs_train, np.ravel(target_train), [1,2,3,4,5,6,7])
	else:
		clf.fit(inputs_train, np.ravel(target_train))
	prediction = clf.predict(inputs_valid)
	correct = np.count_nonzero(np.ravel(target_valid) == prediction)
	total, _ = target_valid.shape
	print name, ": ", float(correct)/total
	print "-"*71

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	print "-"*71
	GaussianNaiveBayes()
	GaussianNaiveBayes(True)
	MultinomialNaiveBayes()
	MultinomialNaiveBayes(True)
	BernoulliNaiveBayes()
	BernoulliNaiveBayes(True)
