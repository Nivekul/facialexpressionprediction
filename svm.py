import numpy as np
from sklearn import svm

def linearSVM(inputs_train, inputs_valid, target_train, target_valid):
	name = "Linear SVM"
	clf = svm.LinearSVC()
	clf.fit(inputs_train, np.ravel(target_train))
	prediction = clf.predict(inputs_valid)
	correct = np.count_nonzero(np.ravel(target_valid) == prediction)
	total = target_valid.shape[0]
	correctRate = (float(correct)/total)*100

	return name, correctRate

if __name__ == '__main__':
	print "-"*71
	linear()
