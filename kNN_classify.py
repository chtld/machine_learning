import numpy as np
from collections import Counter

def kNN_classify(k, X_train, Y_train, x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == Y_train.shape[0], \
        'the number of traindata and label must be same'
    assert x.shape[0] == X_train.shape[1], "the featrue number of x must equal X_train"
    
    distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in X_train]
    nearest = np.argsort(distances)
    
    topk_y = Y_train[nearest[: k]]
    votes = Counter(topk_y)
    return votes.most_common(1)[0][0]