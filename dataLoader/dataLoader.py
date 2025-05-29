'''
written by Lorenz Muller
'''

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

    print('loaded dense data matrix')
    density = np.count_nonzero(trainRatings) / (trainRatings.shape[0] * trainRatings.shape[1])
    print("density: " + str(density))

    return trainRatings, validRatings


def load_jester_data_xls(file_path, valfrac=0.1, seed=1234, transpose=False):
    """
        The `load_jester_data_xls` function prepares data for a machine learning model
        by creating matrix of user ratings for each joke. The dataset can be downloaded from:
        https://goldberg.berkeley.edu/jester-data/

        ### Parameters:
        - `file_path` (str): Path to the dataset directory.
        - `valfrac` (float): Fraction of the dataset to be used for validation (default: 0.1).
        - `seed` (int): Random seed for reproducibility (default: 1234).
        - `transpose` (bool): Not used in this implementation, but can be set for future modifications.

        ### Returns:
        - `train_ratings` (numpy.ndarray): Training set containing movie ratings.
        - `valid_ratings` (numpy.ndarray): Validation set containing movie ratings.
        """

    np.random.seed(seed)

    tic = time()
    print('reading data...')
    df = pd.read_excel(file_path, header=None)

    # First column representing number of ratings per user get deleted
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
    density = np.count_nonzero(train_ratings) / (train_ratings.shape[0] * train_ratings.shape[1])
    print("density: " + str(density))

    return train_ratings, valid_ratings


def load_ratings_with_personality_traits(path='./', valfrac=0.1, seed=1234, feature_classification = False, transpose=False):
    """
       The `load_ratings_with_personality_traits` function prepares data for a machine learning model
       by combining users' movie ratings with their personality traits. The dataset can be downloaded from:
       https://grouplens.org/datasets/personality-2018/

       ### Parameters:
       - `path` (str): Path to the dataset directory.
       - `valfrac` (float): Fraction of the dataset to be used for validation (default: 0.1).
       - `seed` (int): Random seed for reproducibility (default: 1234).
       - `feature_classification` (bool): returns matrix of original shape and converts them into table of vectors
       - `transpose` (bool): Transposes all matrixes.

       ### Returns:
       - `train_ratings` (numpy.ndarray): Training set containing movie ratings.
       - `val_ratings` (numpy.ndarray): Validation set containing movie ratings.
       - `train_user_features` (numpy.ndarray): Training set containing personality traits.
       - `val_user_features` (numpy.ndarray): Validation set containing personality traits.
       """
    np.random.seed(seed)

    df_personality = pd.read_csv(path + "/personality-data.csv")
    df_personality.columns = df_personality.columns.str.strip()
    df_personality = df_personality[["userid", "openness", "agreeableness",
                                     "emotional_stability", "conscientiousness", "extraversion"]]
    df_personality.rename(columns={"userid": "user_id"}, inplace=True)


    df_ratings = pd.read_csv(path + "ratings.csv")
    df_ratings.rename(columns={"useri": "user_id"}, inplace=True)
    df_ratings["user_id"] = df_ratings["user_id"].astype(str)
    df_ratings.columns = df_ratings.columns.str.strip()

    common_users = set(df_ratings["user_id"]).intersection(set(df_personality["user_id"]))
    mun_users = len(common_users)
    num_movies = np.unique(df_ratings["movie_id"]).shape[0]
    num_features = df_personality.shape[1] - 1

    print("number of users: " + str(mun_users))
    print("number of movies: " + str(num_movies))
    print("number of user features: " + str(num_features))

    df = df_ratings.drop_duplicates(subset=["user_id", "movie_id"], keep="first")
    df_pivot = df.pivot(index="user_id", columns="movie_id", values="rating")
    df_pivot.reset_index(inplace=True)

    i = 0
    data = []
    for user in common_users:

        if user not in df_pivot["user_id"].values or user not in df_personality["user_id"].values:
            continue
        else:
            user_ratings = df_pivot[df_pivot["user_id"] == user].iloc[:, 1:].to_numpy()
            user_ratings = np.nan_to_num(user_ratings, nan=0).astype(float)

            user_feature_row = df_personality[df_personality["user_id"] == user].iloc[0, 1:].to_numpy()
            user_feature_row = np.nan_to_num(user_feature_row, nan=0).astype(float)

            row = {
                "user_id": user,
                "ratings": user_ratings.tolist(),
                "features": user_feature_row.tolist()
            }
            data.append(row)

        i += 1

    ratings = np.array([user["ratings"] for user in data], dtype=np.float32).squeeze()
    user_features = np.array([user["features"] for user in data], dtype=np.float32)

    np.random.shuffle(data)
    num_val = int(len(data) * valfrac)

    train_user_features = user_features[num_val:]
    val_user_features = user_features[:num_val]

    train_ratings = ratings[num_val:]
    val_ratings = ratings[:num_val]

    if not feature_classification:
        # Ensuring valid and train matrix are the same shape
        max_rows = max(train_ratings.shape[0], val_ratings.shape[0])
        if train_ratings.shape[0] < max_rows:
            padding = np.zeros((max_rows - train_ratings.shape[0], train_ratings.shape[1]), dtype=np.float32)
            train_ratings = np.vstack([train_ratings, padding])
        if val_ratings.shape[0] < max_rows:
            padding = np.zeros((max_rows - val_ratings.shape[0], val_ratings.shape[1]), dtype=np.float32)
            val_ratings = np.vstack([val_ratings, padding])

        max_rows_features = max(train_user_features.shape[0], val_user_features.shape[0])
        if train_user_features.shape[0] < max_rows_features:
            padding = np.zeros((max_rows_features - train_user_features.shape[0], train_user_features.shape[1]),
                               dtype=np.float32)
            train_user_features = np.vstack([train_user_features, padding])
        if val_user_features.shape[0] < max_rows_features:
            padding = np.zeros((max_rows_features - val_user_features.shape[0], val_user_features.shape[1]),
                               dtype=np.float32)
            val_user_features = np.vstack([val_user_features, padding])
        density = np.count_nonzero(train_ratings) / (train_ratings.shape[0] * train_ratings.shape[1])
        print("density: " + str(density))
    else:
        train_user_features = train_user_features.reshape(train_user_features.shape[0], 1, train_user_features.shape[1])
        val_user_features = val_user_features.reshape(val_user_features.shape[0], 1, val_user_features.shape[1])

        train_ratings = train_ratings.reshape(train_ratings.shape[0], 1, train_ratings.shape[1])
        val_ratings = val_ratings.reshape(val_ratings.shape[0], 1, val_ratings.shape[1])

    if transpose:
        train_ratings = train_ratings.T
        val_ratings = val_ratings.T
        train_user_features = train_user_features.T
        val_user_features = val_user_features.T


    return train_ratings, val_ratings, train_user_features, val_user_features


def load_top_movies_with_personality_traits(path='./', valfrac=0.1, seed=1234, feature_classification = False, transpose=False, n=500):
    """
       The `load_top_movies_with_personality_traits` loads users, their ratings for top n movies and users' personality traits (see load_ratings_with_personality_traits) 

       ### Parameters:
       - `path` (str): Path to the dataset directory.
       - `valfrac` (float): Fraction of the dataset to be used for validation (default: 0.1).
       - `seed` (int): Random seed for reproducibility (default: 1234).
       - `feature_classification` (bool): returns matrix of original shape and converts them into table of vectors
       - `transpose` (bool): Transposes all matrixes.
       - `n` (int): number of top movies to include in the data (default: 500)

       ### Returns:
       - `train_ratings` (numpy.ndarray): Training set containing movie ratings.
       - `val_ratings` (numpy.ndarray): Validation set containing movie ratings.
       - `train_user_features` (numpy.ndarray): Training set containing personality traits.
       - `val_user_features` (numpy.ndarray): Validation set containing personality traits.
       - "top_indices" (numpy.ndarray): list of indices which correspond to movie indices in the original dataset 
       """
    X_train, X_test, p_train, p_test = load_ratings_with_personality_traits(path, valfrac, seed, feature_classification, transpose)
    movie_ratings = np.count_nonzero(X_train, axis=0) + np.count_nonzero(X_test, axis=0)
    movie_ratings = movie_ratings.ravel()
    top_indices = np.argsort(movie_ratings, stable=True)[::-1][:n]
    X_train=X_train.squeeze()
    X_test = X_test.squeeze()
    X_train = X_train[:,top_indices]
    X_test = X_test[:,top_indices]
    X_train = X_train[np.where(np.count_nonzero(X_train, axis=1)>0)]
    p_train = p_train[np.where(np.count_nonzero(X_train, axis=1)>0)]
    X_test = X_test[np.where(np.count_nonzero(X_test, axis=1)>0)]
    p_test = p_test[np.where(np.count_nonzero(X_test, axis=1)>0)]
    return X_train, X_test, p_train, p_test, top_indices

def load_mid_movies_with_personality_traits(path='./', valfrac=0.1, seed=1234, feature_classification = False, transpose=False, n=500, cutoff=10000):
    X_train, X_test, p_train, p_test = load_ratings_with_personality_traits(path, valfrac, seed, feature_classification, transpose)
    movie_ratings = np.count_nonzero(X_train, axis=0) + np.count_nonzero(X_test, axis=0)
    movie_ratings = movie_ratings.ravel()
    indices = np.argsort(movie_ratings, stable=True)[::-1][n:cutoff]
    X_train=X_train.squeeze()
    X_test = X_test.squeeze()
    X_train = X_train[:,indices]
    X_test = X_test[:,indices]
    return X_train, X_test, p_train, p_test, indices

