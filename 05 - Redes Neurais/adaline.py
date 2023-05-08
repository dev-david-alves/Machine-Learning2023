import numpy as np #Biblioteca de manipulacao de arrays Numpy
###Construindo Adaline
class Adaline(object):
    def __init__(self, eta = 0.001, epoch = 100):
        self.eta = eta
        self.epoch = epoch

    def fit(self, X, y):
        np.random.seed(16)
        self.weight_ = np.random.uniform(-1, 1, X.shape[1] + 1)
        self.error_ = []
        
        cost = 0
        for _ in range(self.epoch):
            
            output = self.activation_function(X)
            error = y - output
            
            self.weight_[0] += self.eta * sum(error)
            self.weight_[1:] += self.eta * X.T.dot(error)
            
            cost = 1./2 * sum((error**2))
            self.error_.append(cost)
            
        return self

    def net_input(self, X):
        """Calculo da entrada z"""
        return np.dot(X, self.weight_[1:]) + self.weight_[0]
    def activation_function(self, X):
        """Calculo da saida da funcao g(z)"""
        return self.net_input(X)
    def predict(self, X):
        """Retornar valores binaros 0 ou 1"""
        return np.where(self.activation_function(X) >= 0.0, 1, -1)