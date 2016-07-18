import tensorflow as tf
import maxKern as maxKern
import maxKL as KL
import numpy as np
import maxKernStats as kstat
import maxHelp as Help
import matplotlib.pylab as plt
import maxTypes as ty
import time

tol =1e-3

def link(prev,post):
    assert prev.outd==post.ind
    prev.post=post
    post.prev=prev


class basicLayer(object):
    def __init__(self, l, ind, outd, n, m):
        self.l=l
        self.ind=ind; self.outd=outd
        self.n=n; self.m=m

        self.f_beta_sqrt=tf.Variable(tf.ones([1],dtype=tf.float64),dtype=tf.float64)
        self.f_beta=tf.square(self.f_beta_sqrt)

        self.prev=None; self.post=None

        kern = dict()
        for i in xrange(0, outd):
            kern[i] = maxKern.RBF(ind)
        self.kern=kern



class notLastLayer(basicLayer):
    def __init__(self, l, ind, outd, n, m):
        basicLayer.__init__(self, l, ind, outd, n, m)
        self.ps_P=ty.points(num=m,dim=outd)
        self.M=ty.mean(num=n,dim=outd)
        self.S=ty.diag_cov(num=n,dim=outd)

    def psi(self,d):
        return kstat.build_psi_stats_rbf(self.ps_P.getNxD(), self.post.kern[d],self.M.getNxD(), self.S.getNxD())

    def S_term(self):
            return 0.5*tf.reduce_sum(tf.log(self.S.getNxD()))+(np.math.log(2*np.pi)+1)*self.outd

#####################################################################

class notFstLayer(basicLayer):
    def __init__(self, l, ind, outd, n, m):
        basicLayer.__init__(self, l, ind, outd, n, m)
        self.mu=ty.mu(num=m,dim=outd)
        self.sigma=ty.full_cov(ind=outd,squareDim=m)

    def pre_L_term(self):
        # 1st half of Line 1
        result=0.5*tf.log(2*np.pi*tf.inv(self.f_beta))*self.n*self.outd
        #Line 2 and 3
        for j in xrange(self.outd):
            # Definitions
            zeta_l, psi_l, phi_l=self.prev.psi(j)
            mu_j=self.mu.getDim(j,[-1,1])
            M_j=self.M.getDim(j,[1,-1])
            Kuu_chol=tf.cholesky(self.Kuu(j))

            #term (3)
            result-=self.f_beta*Help.Mul(M_j,psi_l,tf.cholesky_solve(Kuu_chol,mu_j))
            #term (4)
            phi_lT=tf.transpose(phi_l)
            result-=0.5*self.f_beta*(zeta_l-tf.trace(tf.cholesky_solve(Kuu_chol,phi_lT)))

            #term (5)
            mid=tf.matmul(mu_j, tf.transpose(mu_j))+self.sigma.get_dim(j)
            left=tf.cholesky_solve(Kuu_chol, mid)
            trace=tf.trace(tf.matmul(left, tf.cholesky_solve(Kuu_chol, phi_l)))
            result-=0.5*self.f_beta*trace
        return result

 ###############################################

    def Kuu(self,i):
        return self.kern[i].K(self.prev.ps_P.getNxD()) +tol*np.eye(self.m)

    def KL(self):
        KL_div=0
        for d in xrange(0,self.outd):
            sigma_d=tf.expand_dims(self.sigma.get_dim(d),-1)
            mu_d= self.mu.getDim(d,[-1,1])
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
            return KL.group_gauss_kl(self.M.getDxM(),self.S.getDxM(),cov,self.n)

    def bound(self):
        return -self.KL_H()+self.S_term()

class midLayer(notLastLayer, notFstLayer):
    def __init__(self, l, ind, outd, n, m):
        notLastLayer.__init__(self, l, ind, outd, n, m)
        notFstLayer.__init__(self, l, ind, outd, n, m)
    def L_term(self):
        result=0.5*self.f_beta*(tf.reduce_sum(tf.square(self.M.getNxD()))+tf.reduce_sum(self.S.getNxD()))
        result+=self.pre_L_term()
        return result
    def bound(self):
        return self.L_term()-self.KL()+self.S_term()

class lastLayer(notFstLayer):
    def __init__(self, l, ind, outd, n, m,data=None):
        notFstLayer.__init__(self,l,ind,outd,n,m)
        if data==None:
            data= tf.constant(np.random.randn(n, outd),dtype=tf.float64)
        self.data = data
        self.M=ty.mean(num=n, dim=outd,data=self.data)

    def L_term(self):
        result=0.5*self.f_beta*(tf.reduce_sum(tf.square(self.M.getNxD())))
        result+=self.pre_L_term()
        return result

    def bound(self):
        return self.L_term()-self.KL()




def test():
    global s
    s=tf.Session()
    n=20
    layer1=fstLayer(l=1,ind=5,outd=4,n=n,m=5)
    layer2=midLayer(l=1,ind=4,outd=2,n=n,m=5)
    layer3=lastLayer(l=1,ind=2,outd=1,n=n,m=5)
    link(layer1,layer2);  link(layer2,layer3)
    bound=layer1.bound()+layer2.bound()+layer3.bound()
    s.run(tf.initialize_all_variables())
    plt.ion()
    print(s.run(bound))

    opt = tf.train.AdamOptimizer(0.1)
    train=opt.minimize(-1*bound)
    s.run(tf.initialize_all_variables())
    print('begin optimisation')
    count=0
    for step in xrange(10000):
        s.run(train)
        if step % 1 == 0:
            count+=1
            time.sleep(0)
            print(s.run(layer2.Kuu(1)))
            #print('ps',s.run(layer2.ps_P) )
            print(s.run(bound))
        print('count: ', count)


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