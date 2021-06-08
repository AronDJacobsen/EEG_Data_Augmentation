import numpy as np
from sklearn.linear_model import LogisticRegression
from models.models import models

def norm_grad_x_LR(theta, x, C=1.):
    # Function taken from course in Active Machine Learning at DTU - course number: 02463

    # probability of high class
    p1 = 1 / (1 + np.exp(-np.sum(theta * x)))
    # probability of low class
    p0 = 1 - p1
    # derivative of cost as derived above for high class
    dL1 = -x / (1 + np.exp(theta * x)) + theta / C
    # same for low class
    dL0 = x * np.exp(theta * x) / (1 + np.exp(theta * x)) + theta / C
    # 2-norm of these
    g1 = np.sqrt(np.sum(dL1 ** 2, 1))
    g0 = np.sqrt(np.sum(dL0 ** 2, 1))
    # averaged according to probabilities
    emc = p1 * g1 + p0 * g0

    return emc

class ActiveModels:

    def __init__(self, X_train,y_train, X_test, y_test, state):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = ["absent", "present"] # "shiv", "elpp", "musc", "null"]

        self.model = None

        self.state = state

        super(ActiveModels, self)

    def LR(self, C):
        self.model = LogisticRegression(C = C, max_iter = 500)
        model = self.model
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        weights = model.coef_

        accuracy, f2_s, sensitivity = models.scores(self, y_pred)


        return accuracy, f2_s, sensitivity, y_pred, weights