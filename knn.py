import numpy as np
from sklearn import neighbors
from util import *

def main():
	inputs_train, inputs_valid, target_train, target_valid = LoadData('labeled_images.mat', 0.5)
	for k in xrange(2,11,2):
		knn = neighbors.KNeighborsClassifier(k)
		knn.fit(inputs_train, np.ravel(target_train))
		prediction = knn.predict(inputs_valid)
		correct = np.count_nonzero(np.ravel(target_valid) == prediction)
		total, _ = target_valid.shape
		print k, float(correct)/total

if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	main()
