import tensorflow as tf
import numpy as np

# Abstractly, this class represents a kernel function
class Kern:

    def __init__(self, input_dim, variance=1.0, lengthscales=None):
        self.input_dim=input_dim; self.variance=variance
        if lengthscales is None:
            lengthscales=np.ones(input_dim)
        self.lengthscales=lengthscales

    def square_dist(self, X, X2):
        X = X/self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2*tf.matmul(X, tf.transpose(X)) +\
                tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2*tf.matmul(X, tf.transpose(X2)) +\
                tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def K(self, X, X2=None):
        return self.variance * tf.exp(-self.square_dist(X, X2)/2)