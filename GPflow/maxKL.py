import tensorflow as tf
from tf_hacks import eye
#import maxGPLVM as master
import maxHelp as h
import numpy as np
import maxTypes
s=None

def setSession(x):
    global s
    s=x


def group_gauss_kl(q_mu, q_sq,k, num_latent):
    # K is a variance...we sqrt later
    assert(h.get_dim(q_mu,0)==h.get_dim(q_sq,0))
    assert(h.get_dim(q_mu,1)==h.get_dim(q_sq,1))
    assert(h.get_dim(q_mu,1)==num_latent)
    assert(h.get_dim(q_mu,0)==h.get_dim(k,0))
    k=tf.reshape(k,[-1,1])
    assert(h.get_dim(q_mu,0)==h.get_dim(k,0))

    """
    Compute the KL divergence KL(q||p)

          q(x) = N(q_mu, q_sq)

          p(x) = N(0, I)

    We assume num_latent independent distributions, given by the columns of
    q_mu and q_sq=(q_sqrt)^2

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each column represents the diagonal of a square-root
        matrix of the covariance.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and q_sqrt).
    """

    KL=0.5*tf.reduce_sum(tf.log(k/q_sq))
    KL+=0.5*tf.reduce_sum(tf.square(q_mu)/k)
    KL+=0.5*tf.reduce_sum(q_sq/k)
    KL-= 0.5 * tf.cast(tf.shape(q_sq)[0] * num_latent, tf.float64) #stay same
    return KL

# TODO Fix ME
def gauss_kl(min_q_mu, q_sq,K):
    q_mu=-1*min_q_mu

    #q_sqrt=tf.cholesky(tf.squeeze(q_sqrt))
        # K is a variance...we sqrt later
    '''
    N=1
    Q=5
    q_mu=tf.random_normal([Q,1],dtype=tf.float64)
    q_var=tf.random_normal([Q,Q],dtype=tf.float64)
    q_var=q_var+tf.transpose(q_var [1,0])+1e+1*np.eye(Q)
    K=q_var
    q_sqrt=tf.cholesky(q_var)
    q_sqrt=tf.expand_dims(q_sqrt,-1)
    num_latent=1
    s=tf.Session()
    s.run(tf.initialize_all_variables())
    '''
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume num_latent independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean.

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and the last dim of q_sqrt).

    q_sqrt=tf.cholesky(K)
    L = tf.cholesky(q_sq)
    alpha = tf.matrix_triangular_solve(L, q_mu, lower=True)
    KL = 0.5 * tf.reduce_sum(tf.square(alpha))  # Mahalanobis term.
    KL +=   0.5 * tf.reduce_sum(
        tf.log(tf.square(tf.diag_part(L))))  # Prior log-det term.
    KL += -0.5 * tf.cast(tf.shape(q_sqrt)[0], tf.float64)

    Lq = tf.batch_matrix_band_part(q_sqrt, -1, 0)
    # Log determinant of q covariance:
    KL += -0.5*tf.reduce_sum(tf.log(tf.square(tf.diag_part(Lq))))
    LiLq = tf.matrix_triangular_solve(L, Lq, lower=True)
    KL += 0.5 * tf.reduce_sum(tf.square(LiLq))  # Trace term
    """
    V2=tf.cholesky(K)
    V1=tf.cholesky(q_sq)
    KL=h.Mul(tf.transpose(q_mu),tf.cholesky_solve(V2,q_mu))
    KL+=tf.trace(tf.cholesky_solve(V2,q_sq))
    KL-=h.get_dim(K,0)
    KL+=tf.reduce_sum(2*tf.log(tf.diag_part(V2))-2*tf.log(tf.diag_part(V1)))
    return KL/2


