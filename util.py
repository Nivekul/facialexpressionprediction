import numpy as np
import scipy.io as sp
from a3 import *
from sklearn.cross_validation import LabelKFold
from sklearn import preprocessing

def testRunLoadData(n_fold=3):
    filename = "labeled_images.mat"
    image_data, labels, identity = loadMatFile(filename)
    label_kfold = LabelKFold(identity, n_folds = n_fold)
    return label_kfold


def LoadData(filename, train_portion = 0.25, normalize=False):
    data = sp.loadmat(filename)
    inputs = data['tr_images']
    target = data['tr_labels']
    identities = data['tr_identity']
    if (train_portion == 0):
        return inputs, target

    size = target.size
    i = size*train_portion

    unknown = np.argwhere(identities == -1)[:,0].size
    identities_idx = np.argsort(identities, 0)
    unknown_idx = np.ravel(identities_idx[:unknown])
    identities_idx = np.ravel(identities_idx[unknown+1:])


    j = identities_idx.size*train_portion
    people_train, people_valid = np.split(identities_idx, [j])

    inputs = np.reshape(inputs.T, (size, 1024))
    if normalize: inputs = preprocessing.scale(inputs)
    index_train, index_valid = np.split(np.random.permutation(size), [i])

    inputs_train = inputs[people_train]
    inputs_valid = inputs[people_valid]
    target_train = target[people_train]
    target_valid = target[people_valid]
    return inputs_train, inputs_valid, target_train, target_valid


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    LoadData('labeled_images.mat')
