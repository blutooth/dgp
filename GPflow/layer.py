import tensorflow as tf
import maxKern as maxKern
import maxKL as KL
import numpy as np
import maxKernStats as kstat
import maxHelp as Help
def link(prev,post):
    assert prev.outd==post.ind
    prev.post=post
    post.prev=prev


class basicLayer(object):
    def __init__(self, l, ind, outd, n, m):
        self.l=l
        self.ind=ind
        self.outd=outd
        self.n=n
        self.m=m
        self.f_beta_sqrt=tf.Variable(tf.ones([1],dtype=tf.float64),dtype=tf.float64)
        self.f_beta=tf.square(self.f_beta_sqrt)
        self.prev=None
        self.post=None
        kern = dict()
        for i in xrange(0, outd):
            kern[i] = maxKern.RBF(ind)
        self.kern=kern



class notLastLayer(basicLayer):
    def __init__(self, l, ind, outd, n, m):
        basicLayer.__init__(self, l, ind, outd, n, m)
        self.ps_P=tf.Variable(tf.ones([m,outd],dtype=tf.float64),dtype=tf.float64)
        self.M=tf.Variable(tf.ones([n,outd],dtype=tf.float64),dtype=tf.float64)
        # S is not squared
        self.S=tf.Variable(tf.ones([n,outd],dtype=tf.float64),dtype=tf.float64)
        self.S_squared=tf.square(self.S)

    def psi(self,d):
        return kstat.build_psi_stats_rbf(self.ps_P, self.post.kern[d],self.M, self.S)

    def S_term(self):
            return tf.reduce_sum(tf.log(self.S_squared))

class notFstLayer(basicLayer):
    def __init__(self, l, ind, outd, n, m):
        basicLayer.__init__(self, l, ind, outd, n, m)
        self.mu=tf.Variable(tf.ones([outd,m],dtype=tf.float64),dtype=tf.float64)
        self.sigma=tf.Variable(tf.ones([outd,m,m],dtype=tf.float64),dtype=tf.float64)

    def L_term(self):

        #term (1)
        result=-0.5*tf.log(2*np.pi*tf.inv(self.f_beta))*self.n*self.outd
        #term (2)
        result+=0.5*self.f_beta*(tf.reduce_sum(tf.square(self.M))+tf.reduce_sum(self.S))
        #term (3)
        term3=0
        # TODO Get Statistics only once per j
        for j in xrange(self.outd):
            zeta_l, psi_l, phi_l=self.prev.psi(j)
            mu_j=tf.reshape(tf.slice(self.mu,[j,0],[1,-1]),[-1,1])
            M_j=tf.reshape(tf.slice(self.M,[0,j],[-1,1]),[1,-1])
            term3=Help.Mul(M_j,psi_l,tf.cholesky_solve(tf.cholesky(self.Kuu(j)),mu_j))
            result-=self.f_beta*term3
            #term (4)
            term4=zeta_l
            phi_lT=tf.transpose(phi_l)
            term4=tf.trace(tf.cholesky_solve(tf.cholesky(self.Kuu(j)),phi_lT))
            result-=0.5*self.f_beta*term4
            #term (5)
            term5=0
            #print(s.run(self.Kuu(j)))
            midTerm=tf.matmul(mu_j, tf.transpose(mu_j))+tf.squeeze(tf.slice(self.sigma,[j,0,0],[1,-1,-1]))
            first=tf.cholesky_solve(tf.cholesky(self.Kuu(j)), midTerm)
            term5=tf.trace(tf.matmul(first, tf.cholesky_solve(tf.cholesky(self.Kuu(j)), phi_l)))
            result-=0.5*self.f_beta*term5
        return result



    def Kuu(self,i):
        return self.kern[i].K(self.prev.ps_P) +1e-4*np.eye(self.m)

    def KL(self):
        KL_div=0
        for d in xrange(0,self.outd):
            sigma_d=tf.expand_dims(tf.squeeze(tf.slice(self.sigma,[d,0,0],[1,-1,-1])),-1)
            mu_d= tf.reshape(tf.slice(self.mu,[d,0],[1,-1]),[-1,1])
            KL_div+=KL.gauss_kl(mu_d, sigma_d, self.Kuu(d), 1)
        return KL_div



class fstLayer(notLastLayer):
    def __init__(self, l, ind, outd, n, m,data=None):
        notLastLayer.__init__(self, l, ind, outd, n, m)
        if data==None:
            data= tf.constant(np.random.randn(n, ind),dtype=tf.float64)
        self.dataPoints = data

    def KL_H(self):
            cov=tf.squeeze(tf.pack([self.kern[d].variance for d in xrange(self.outd)]))
            return KL.gauss_kl_diag_diag(tf.transpose(self.M),tf.transpose(self.S),cov,self.n)

class midLayer(notLastLayer, notFstLayer):
    def __init__(self, l, ind, outd, n, m):
        notLastLayer.__init__(self, l, ind, outd, n, m)
        notFstLayer.__init__(self, l, ind, outd, n, m)


class lastLayer(notFstLayer):
    def __init__(self, l, ind, outd, n, m,data=None):
        notFstLayer.__init__(self,l,ind,outd,n,m)
        if data==None:
            data= tf.constant(np.random.randn(n, outd),dtype=tf.float64)
        self.data = data

    def L_term(self):

        #term (1)
        result=-0.5*tf.log(2*np.pi*tf.inv(self.f_beta))*self.n*self.outd
        #term (2)
        result+=0.5*self.f_beta*tf.reduce_sum(tf.square(self.data))
        #term (3)
        term3=0
        # TODO Get Statistics only once per j
        for j in xrange(self.outd):
            zeta_l, psi_l, phi_l=self.prev.psi(j)
            mu_j=tf.reshape(tf.slice(self.mu,[j,0],[1,-1]),[-1,1])
            M_j=tf.reshape(tf.slice(self.data,[0,j],[-1,1]),[1,-1])
            term3=Help.Mul(M_j,psi_l,tf.cholesky_solve(tf.cholesky(self.Kuu(j)),mu_j))
            result-=self.f_beta*term3
            #term (4)
            term4=zeta_l
            phi_lT=tf.transpose(phi_l)
            term4=tf.trace(tf.cholesky_solve(tf.cholesky(self.Kuu(j)),phi_lT))
            result-=0.5*self.f_beta*term4
            #term (5)
            term5=0
            #print(s.run(self.Kuu(j)))
            midTerm=tf.matmul(mu_j, tf.transpose(mu_j))+tf.squeeze(tf.slice(self.sigma,[j,0,0],[1,-1,-1]))
            first=tf.cholesky_solve(tf.cholesky(self.Kuu(j)), midTerm)
            term5=tf.trace(tf.matmul(first, tf.cholesky_solve(tf.cholesky(self.Kuu(j)), phi_l)))
            result-=0.5*self.f_beta*term5
        return result


def test():
    global s
    s=tf.Session()
    n=20
    layer1=fstLayer(l=1,ind=5,outd=4,n=n,m=5)
    layer2=midLayer(l=1,ind=4,outd=2,n=n,m=5)
    layer3=lastLayer(l=1,ind=2,outd=1,n=n,m=5)
    link(layer1,layer2);  link(layer2,layer3)
    bound=layer1.KL_H()+layer2.L_term()+layer2.KL()+layer2.S_term()+layer3.KL()+layer3.L_term()+layer1.S_term()
    s.run(tf.initialize_all_variables())
    print(s.run(bound))
    '''
    fl=True
    if fl==True:

        firstLayer= fstLayer(l=1,ind=5,outd=4,n=10,m=3)
        s.run(tf.initialize_all_variables())
        print(s.run(firstLayer.KL_H()))

    sl=True
    if sl==True:
        midL= midLayer(l=1,ind=5,outd=4,n=10,m=3)
        s.run(tf.initialize_all_variables())
        print(s.run(midL.Kuu(1)))
        print(s.run(midL.KL()))
        print(s.run(midL.psi(1)))
        #s.run(tf.initialize_all_variables())
        '''

test()