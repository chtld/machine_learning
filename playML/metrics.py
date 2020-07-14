import numpy as np

def accuracy_score(Y_true, Y_predict):
    assert Y_true.shape[0] == Y_predict.shape[0], "the true must have the same number as the predict"

    accuracy = np.sum(Y_predict == Y_true) / len(Y_true)
    return accuracy