__all__ = ["Surrogate"]


class Surrogate(object):
    def __init__(self):
        pass

    def train(self, X_train, Y_train, V_train):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def predict(self, X_data):
        raise NotImplementedError("This method should be implemented in a subclass.")

