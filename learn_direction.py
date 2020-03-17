import os
import pickle
import config
import dnnlib
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

directory = 'results/vm/landscapes-no-cond/generated_imgs1000/'


def main():
    directions = ['water', 'grass', 'outdoor', 'field', 'beach', 'lake', 'ocean', 'painting', 'river', 'tree',
                  'nature', 'night', 'sunset', 'wave', 'blue', 'mountain', 'clouds', 'hill']

    for d in directions:
        learn_direction(d)


def learn_direction(label):
    qlatents = pickle.load(open(directory+'landscape_latents.p', 'rb'))
    dlatents = pickle.load(open(directory+'landscape_dlatents.p', 'rb'))

    X_data = np.asarray(dlatents).reshape((-1, 16*512))

    Y_data = get_y_for_label(label)

    assert(len(X_data) == len(Y_data))

    clf = LogisticRegression(class_weight='balanced').fit(X_data, (Y_data))
    direction = clf.coef_.reshape((16, 512))

    pickle.dump(direction, open(directory+'directions/'+label+'.p', 'wb'))


def get_y_for_label(label):
    y = []
    labels_data = get_labels()
    for x in labels_data:
        if label in x['description']['tags']:
            y.append(1)
        else:
            y.append(0)
    print(label + str(np.sum(y)))
    return y


def get_labels():
    labels_data = []
    for i in range(1000):
        filename = directory + 'analysis/img_' + str(i) + '.p'
        labels_data.append(pickle.load(open(filename, 'rb')))
    return labels_data


if __name__ == "__main__":
    main()
