import tensorflow as tf
import numpy as np
import maxKern as kern
import maxHelp as h
'''
s=None

def getSession(sess):
    global s
    s=sess
'''

def build_psi_stats_rbf(Z, kern, mu, S):
    #Check Consistent Shapes
    assert(kern.input_dim==h.get_dim(mu,1))
    assert(kern.input_dim==h.get_dim(S,1))
    assert(kern.input_dim==h.get_dim(Z,1))
    assert(h.get_dim(mu,0)==h.get_dim(S,0))

    # expecting below note S is the squared covariance

    '''
    Q = 3
    N = 4
    M = 2



    Z=tf.transpose(Z)
    mu=tf.transpose(mu)
    S=tf.transpose(S)
    ### works! Yippee
    mu=tf.random_uniform([N,Q], minval=0, maxval=10, dtype=tf.float64)
    S=tf.random_uniform([N,Q], minval=0, maxval=10, dtype=tf.float64)
    Z=tf.random_uniform([M,Q], minval=0, maxval=10, dtype=tf.float64)
    var=tf.constant(1.0,dtype=tf.float64)
    lengthscale2=tf.constant([3.0,2.0,1.0],dtype=tf.float64)
    '''

    var=kern.variance
    lengthscale2 = tf.square(kern.lengthscales)


    # psi0
    N = tf.shape(mu)[0]
    psi0 = tf.cast(N, tf.float64) * var
    # psi1
    psi1_logdenom = tf.expand_dims(tf.reduce_sum(tf.log(S / lengthscale2 + 1.), 1), 1)  # N x 1
    d = tf.square(tf.expand_dims(mu, 1)-tf.expand_dims(Z, 0))  # N x M x Q
    psi1_log = - 0.5 * (psi1_logdenom + tf.reduce_sum(d/tf.expand_dims(S+lengthscale2, 1), 2))
    psi1 = var * tf.exp(psi1_log)

    # psi2
    psi2_logdenom = -0.5 * tf.expand_dims(tf.reduce_sum(tf.log(2.*S/lengthscale2 + 1.), 1), 1)  # N # 1
    psi2_logdenom = tf.expand_dims(psi2_logdenom, 1)
    psi2_exp1 = 0.25 * tf.reduce_sum(tf.square(tf.expand_dims(Z, 1)-tf.expand_dims(Z, 0))/lengthscale2, 2)  # M x M
    psi2_exp1 = tf.expand_dims(psi2_exp1, 0)

    Z_hat = 0.5 * (tf.expand_dims(Z, 1) + tf.expand_dims(Z, 0))  # MxMxQ
    denom = 1./(2.*S+lengthscale2)
    a = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.square(mu)*denom, 1), 1), 1)  # N x 1 x 1
    b = tf.reduce_sum(tf.expand_dims(tf.expand_dims(denom, 1), 1) * tf.square(Z_hat), 3)  # N M M
    c = -2*tf.reduce_sum(tf.expand_dims(tf.expand_dims(mu*denom, 1), 1) * Z_hat, 3)  # N M M
    psi2_exp2 = a + b + c

    psi2 = tf.square(var) * tf.reduce_sum(tf.exp(psi2_logdenom - psi2_exp1 - psi2_exp2), 0)
    return psi0, psi1, psi2

