import scipy.io as sio
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt; plt.rcdefaults()
from knn import *
from sklearn import cross_validation
from sklearn import svm
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import LabelKFold

from ovo import *
from svm import *
from nb import *

def loadMatFile(filename):
	"""
	load .mat file and return transformed image_data and associated labels
	"""
	mat_contents = sio.loadmat(filename)
	# key = mat_contents.keys()
	data = mat_contents['tr_images'].T
	num_images = mat_contents['tr_images'].shape[2]
	identity = mat_contents['tr_identity']
	image_data = np.reshape(data, (num_images, 1024))
	labels = mat_contents['tr_labels'].T[0]

	return image_data, labels, identity


def generateDataByIndex(train_index, valid_index, image_data, labels):
	"""
	Generate data sets by index

	train_index: np array

	"""
	train_data = image_data[train_index]
	valid_data = image_data[valid_index]
	train_label = labels[train_index]
	valid_label = labels[valid_index]
	return train_data, valid_data, train_label, valid_label


def plotCorrectness(correctness, plot_title):
	"""
	plot keys VS values in correctness
	"""
	x = correctness.keys()
	y = correctness.values()
	plt.xlabel('k-Values')
	plt.ylabel('Correctness')
	plt.title(plot_title)
	plt.ylim(1,100)
	plt.xlim(0,11)
	plt.plot(x, y, ".")
	plt.show()

def addResult(algo, name, correctRate):
	if name in algo:
		_ = algo[name]
		_.append(correctRate)
		algo[name] = _
	else:
		algo[name] = [correctRate]

def printResult(algo):
	print "-"*71
	for name in algo:
		print name, ": ", sum(algo[name])/float(len(algo[name]))
		print "-"*71

if __name__ == '__main__':
	kValue = [2,4,6,8,10]
	# load data from .mat file
	# returns a dictionary with keys
	# ['__globals__', 'tr_labels', '__header__', 'tr_identity', '__version__', 'tr_images']
	image_data, labels, identity = loadMatFile('labeled_images.mat')

	# # ramdom select data from data set as traina and validation sets
	# data_train, data_valid, label_train, label_valid = train_test_split(image_data, labels, test_size=0.33, random_state=42)

	###################################################################
	# identity k fold #
	###################################################################
	corr = []
	c = {}
	lkf = LabelKFold(identity, n_folds=3)
	for k in kValue:
		for train, valid in lkf:
			train_data, valid_data, train_label, valid_label = generateDataByIndex(train, valid, image_data, labels)

			corr.append(knn(k, train_data, train_label, valid_data, valid_label))
		c[k] = sum(corr) / 3
		corr = []
	print "identity-k-fold with knn:", c

	# n_fold = 3
	plot_title = 'identity-k-fold with knn (n_fold = 3)'
	# c = {8: 52.478632478632484, 2: 51.28205128205128, 4: 53.53846153846154, 10: 52.78632478632479, 6: 53.641025641025635}
	plotCorrectness(c, plot_title)

	algo = {}
	for train, valid in lkf:
		train_data, valid_data, train_label, valid_label = generateDataByIndex(train, valid, image_data, labels)

		name, correctRate = OneVsOne(train_data, valid_data, train_label, valid_label)
		addResult(algo, name, correctRate)
		print "Finished " + name
		name, correctRate = linearSVM(train_data, valid_data, train_label, valid_label)
		addResult(algo, name, correctRate)
		print "Finished " + name
		name, correctRate = GaussianNaiveBayes(train_data, valid_data, train_label, valid_label)
		addResult(algo, name, correctRate)
		print "Finished " + name
		name, correctRate = MultinomialNaiveBayes(train_data, valid_data, train_label, valid_label)
		addResult(algo, name, correctRate)
		print "Finished " + name
		name, correctRate = BernoulliNaiveBayes(train_data, valid_data, train_label, valid_label)
		addResult(algo, name, correctRate)
		print "Finished " + name

	printResult(algo)
