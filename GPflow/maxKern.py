import tensorflow as tf
import numpy as np


class Stationary():
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """
    def __init__(self, input_dim, variance=None,lengthscales=None ):
        if variance==None:
            self.variance=tf.Variable(1.0,dtype=tf.float64)
        else:
            self.variance=variance
        if lengthscales==None:
            self.lengthscales=tf.Variable(tf.ones([input_dim],dtype=tf.float64),dtype=tf.float64)
        else:
            self.lengthscales=lengthscales
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        #
        #

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

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        return tf.fill(tf.pack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class RBF(Stationary):

    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def K(self, X, X2=None):
        return self.variance * tf.exp(-self.square_dist(X, X2)/2)

# Abstractly, this class represents a Squared exponential kernel function
# has length-scales

# Abstractly, this class represents a kernel function
class Kern:

    def __init__(self, dim, var=1.0, lenSc=None):
        self.dim=dim; self.var=var
        if lenSc is None:
            lenSc=np.ones(dim)
        self.lenSc=lenSc

    def square_dist(self, X, X2):
        X = X/self.lenSc
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2*tf.matmul(X, tf.transpose(X)) +\
                tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lenSc
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2*tf.matmul(X, tf.transpose(X2)) +\
                tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def K(self, X, X2=None):
        return self.var * tf.exp(-self.square_dist(X, X2)/2)

    def m_square_dist(self,X1, X2):
        K_1 = tf.divide(tf.square(tf.abs(tf.sub(X1, X2))),self.lenSc)
        return K_1

    def m_K(self, X, X2=None):
        return tf.squeeze(self.var * tf.exp(-self.square_dist(X, X2)/2))

    def m_K_all(self,X,X2,n,m):
        mat=[]
        for i in xrange(n):
            row=[]
            for j in xrange(m):
                X_i=tf.slice(X,[i,0],[1,-1])
                X_j=tf.slice(X2,[j,0],[1,-1])
                row+=Kern.K(X_i,X_j)
            mat+=tf.pack(row)
        return tf.pack(mat)
