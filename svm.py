import numpy as np
from util import *
from sklearn import svm

inputs_train, inputs_valid, target_train, target_valid = LoadData('labeled_images.mat', 0.3)

def linear():
	name = "Linear SVM"
	clf = svm.LinearSVC()
	clf.fit(inputs_train, np.ravel(target_train))
	prediction = clf.predict(inputs_valid)
	correct = np.count_nonzero(np.ravel(target_valid) == prediction)
	total, _ = target_valid.shape
	print name, ": ", float(correct)/total
	print "-"*71

if __name__ == '__main__':
	print "-"*71
	linear()
