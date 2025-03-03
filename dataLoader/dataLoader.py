'''
written by Lorenz Muller
'''

import numpy as np
from time import time


def load_data(path='./', valfrac=0.1, trainfrac=0.1, delimiter='::', seed=1234,
             transpose=False):
    '''
    loads ml-1m data

    :param path: path to the ratings file
    :param valfrac: fraction of data to use for validation
    :param delimiter: delimiter used in data file
    :param seed: random seed for validation splitting
    :param transpose: flag to transpose output matrices (swapping users with movies)
    :return: train ratings (n_u, n_m), valid ratings (n_u, n_m)
    '''
    np.random.seed(seed)

    tic = time()
    print('reading data...')
    data = np.genfromtxt(path, delimiter=delimiter, dtype='int32')
    print('data read in', time() - tic, 'seconds')

    n_u = np.unique(data[:, 0]).shape[0]  # number of users
    n_m = np.unique(data[:, 1]).shape[0]  # number of movies
    n_r = data.shape[0]  # number of ratings

    # these dictionaries define a mapping from user/movie id to to user/movie number (contiguous from zero)
    udict = {}
    for i, u in enumerate(np.unique(data[:, 0]).tolist()):
        udict[u] = i
    mdict = {}
    for i, m in enumerate(np.unique(data[:, 1]).tolist()):
        mdict[m] = i

    # shuffle indices
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    trainRatings = np.zeros((n_u, n_m), dtype='float32')
    validRatings = np.zeros((n_u, n_m), dtype='float32')
    testRatings = np.zeros((n_u, n_m), dtype='float32')

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]

        # the first few ratings of the shuffled data array are validation data
        if i <= valfrac * n_r:
            validRatings[udict[u_id], mdict[m_id]] = int(r)
        # the rest are training data
        elif i<=valfrac * n_r + trainfrac * n_r:
            testRatings[udict[u_id], mdict[m_id]] = int(r)
        else:
            trainRatings[udict[u_id], mdict[m_id]] = int(r)

    if transpose:
        trainRatings = trainRatings.T
        validRatings = validRatings.T
        testRatings = testRatings.T

    print('loaded dense data matrix')
    print(trainRatings)

    return trainRatings, testRatings, validRatings

"""
Zwraca miacierz users x movies z oceną uzytkownika danego filmu albo 0 jesli nie ma oceny
"""

