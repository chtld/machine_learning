import numpy as np

def train_test_split(data, target, test_ratio = 0.2, seed = None):
    assert data.shape[0] == target.shape[0], "the number of data must same as target"
    assert 0.0 <= test_ratio <= 1.0, "test_ratio must be invalid"

    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indexes = shuffle_indexes[: test_size]
    train_indexes = shuffle_indexes[test_size :]
    X_train = data[train_indexes]
    Y_train = target[train_indexes]
    X_test = data[test_indexes]
    Y_test = target[test_indexes]
    return X_train, X_test, Y_train, Y_test