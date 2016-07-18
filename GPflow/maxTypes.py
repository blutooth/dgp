import tensorflow as tf
import numpy as np
import maxHelp as Help

tol=1e-2

class numxdim:
    def __init__(self, num, dim, data=None):
        self.num=num
        self.dim=dim
        if data == None:
            self.data=tf.Variable(tf.ones([num,dim],dtype=tf.float64),dtype=tf.float64)
        else:
            assert Help.get_dim(data,0)==num
            assert Help.get_dim(data,1)==dim
            self.data=data

    def getDim(self, j,a=[-1, 1]):
        #a=[-1] flattens
        assert j < self.dim
        return tf.reshape(tf.slice(self.data, [0, j], [-1, 1]), a)

    def getN(self,n,a=[-1, 1]):
        assert n < self.num
        return tf.reshape(tf.slice(self.data, [n, 0], [1, -1]), a)

    def getDxM(self):
        return tf.transpose(self.data)

    def getNxD(self):
        return self.data

class mean(numxdim):
    def __init__(self, num, dim, data=None):
        numxdim.__init__(self, num, dim, data)
        self.myType = "h_m"
        self.type = "mean"

class mu(numxdim):
    def __init__(self, num, dim, data=None):
        numxdim.__init__(self, num, dim, data)
        self.myType = "u_m"
        self.type = "mean"


class diag_cov(numxdim):
    def __init__(self, num, dim, data=None):
        numxdim.__init__(self, num, dim, data)
        self.myType="h_c"
        self.type = "diag"
        self.data=tf.square(self.data)

class full_cov:
    # for each outd (dim1), we have a square matrix (dim2xdim2)
    def __init__(self, ind, squareDim, data=None):
        self.ind=ind
        self.squareDim=squareDim
        if data == None:
            data=[]
            for d in xrange(ind):
                data.append(self.Sigma(squareDim))
            self.data=tf.pack(data,0)
        else:
            assert Help.get_dim(data, 0) == ind
            assert Help.get_dim(data, 1) == squareDim
            assert Help.get_dim(data, 2) == squareDim
            self.data=data

        self.myType="u_c"
        self.type="full"

    def Sigma(self,m):
        id=tol * np.eye(m, dtype=np.float64)
        A=tf.Variable(tf.ones([m, m], dtype=tf.float64), dtype=tf.float64)
        return id + tf.matmul(A, tf.transpose(A))

    def get_dim(self, j):
        assert j < self.ind
        return tf.squeeze(tf.slice(self.data,[j,0,0],[1,-1,-1]))



class points(numxdim):
    def __init__(self, num, dim, data=None):
        numxdim.__init__(self, num, dim, data)
        self.myType = "ps_p"
        self.type = "point"
