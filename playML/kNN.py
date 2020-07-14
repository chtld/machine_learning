import numpy as np
from collections import Counter
from metrics import accuracy_score

class kNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._Y_train = None
    
    def fit(self, X_train, Y_train):
        assert X_train.shape[0] == Y_train.shape[0], \
            "the number of data and label must be same"
        assert self.k <= X_train.shape[0], "the k must less than the data number"
        self._X_train = X_train
        self._Y_train = Y_train
        return self
    
    def predict(self, X_predict):
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature of predict must equal train"
        Y_predict = [self._predict(x) for x in X_predict]   
        return np.array(Y_predict)
    
    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topk_y = self._Y_train[nearest[: self.k]]
        votes = Counter(topk_y)
        return votes.most_common(1)[0][0]
    
    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predict)
        return accuracy

    def __repr__(self):
        return "KNN(k = %d)" % self.k