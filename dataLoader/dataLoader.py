from io import StringIO
import numpy as np
from time import time
import pandas as pd
import xlrd # usage is hidden but must be installed to read xls


def load_movie_lens(path='./', valfrac=0.1, delimiter='::', seed=1234,
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
    new_delim = ':'
    s = open(path).read().replace(delimiter, new_delim)
    data = np.loadtxt(StringIO(s), skiprows=0, delimiter=new_delim).astype('int32')
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

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]

        # the first few ratings of the shuffled data array are validation data
        if i <= valfrac * n_r:
            validRatings[udict[u_id], mdict[m_id]] = int(r)
        # the rest are training data
        else:
            trainRatings[udict[u_id], mdict[m_id]] = int(r)

    if transpose:
        trainRatings = trainRatings.T
        validRatings = validRatings.T

    print("train samples" + str(len(trainRatings)))
    print("validation samples" + str(len(validRatings)))

    print('loaded dense data matrix')

    return trainRatings, validRatings

def load_jester_data_xls(file_path, valfrac=0.1, seed=1234, transpose=False):
    np.random.seed(seed)

    tic = time()
    print('reading data...')
    df = pd.read_excel(file_path, header=None)

    # First column represents number of ratings per user
    num_ratings = df.iloc[:, 0].sum()
    df = df.drop(columns=[0])
    print('data read in', time() - tic, 'seconds')

    matrix = df.to_numpy().astype(np.float32)
    num_users, num_jokes = matrix.shape
    print(f"number of users: {num_users}") # rows
    print(f"number of jokes: {num_jokes}") # columns
    print(f"number of ratings: {num_ratings}") # sum of first column

    # rating normalization
    # Original range: (-10.00, 10.00) and 99 means no rating
    # fixed range (1.00, 21.00) and 0 means no rating
    matrix += 11
    matrix[matrix == 110] = 0

    idx = np.arange(num_users)
    np.random.shuffle(idx)

    num_val = int(valfrac * num_users)

    train_ratings = matrix[idx[num_val:], :]
    valid_ratings = matrix[idx[:num_val], :]
    print("train samples" + str(len(train_ratings)))
    print("validation samples " + str(len(valid_ratings)))


    # Ensuring valid and train matrix are the same shape
    max_users = max(train_ratings.shape[0], valid_ratings.shape[0])
    train_ratings = np.pad(train_ratings, ((0, max_users - train_ratings.shape[0]), (0, 0)), mode='constant')
    valid_ratings = np.pad(valid_ratings, ((0, max_users - valid_ratings.shape[0]), (0, 0)), mode='constant')

    if transpose:
        train_ratings = train_ratings.T
        valid_ratings = valid_ratings.T

    print('Loaded dense data matrix')

    return train_ratings, valid_ratings
