import tensorflow as tf
import tf_hacks as tfh
import maxKern as Kern

class Param:
    def __init__(self,n,m,d,l):
        self.D=dict()


class Model_Param(Param):

        def __init__(s,n,m,d,l):
            s.D=dict()
            s.D['f_noise']=tf.Variable(tf.ones([1.0,l],dtype=tf.float64),dtype=tf.float64)
            s.D['f_beta']=tf.square(s.D['f_noise'])
            s.D['K_std']=tf.Variable(tf.ones([d,l],dtype=tf.float64),dtype=tf.float64)
            s.D['K_var']=tf.square(s.D['K_std'])
            s.D['K_lenSc']=tf.Variable(tf.ones([d,d,l],dtype=tf.float64),dtype=tf.float64)
            for L in xrange(1,l+1):
                for D in xrange(1,d+1):
                    s.D['Kernel',L,D]=Kern.Kern(d,s.get('K_var',l,d),s.get('K_lenSc',l,d))

        def get(s,name,l,d=None):
            if name=='K_lenSc' and d!=None:
                 return tf.squeeze(tf.slice(s.D[name],[0,d-1,l-1],[-1,1,1]))
            if d==None and l!=None:
                return tf.squeeze(tf.slice(s.D[name],[0,l-1],[0,1]))
            elif d!=None and l!=None:
                return tf.squeeze(tf.slice(s.D[name],[d-1,l-1],[1,1]))
            print('slicing not implemented')





class Var_Param(Param):
    def __init__(self,n,m,d,l):
        self.D=dict()
        self.D['pseudo_Sigma_param']=tf.Variable(tf.random_normal([m,m,d,l], dtype=tf.float64),dtype=tf.float64)
        self.D['pseudo_Sigma']=self.D['pseudo_Sigma_param']+tf.transpose(self.D['pseudo_Sigma_param'],[1,0,2,3])
        self.D['pseudo_mu']=tf.Variable(tf.ones([m,d,l],dtype=tf.float64),dtype=tf.float64)
        self.D['point_mu']=tf.Variable(tf.ones([d,n,l],dtype=tf.float64),dtype=tf.float64)
        self.D['point_Sigma']=tf.Variable(tf.random_normal([d,n,l],dtype=tf.float64),dtype=tf.float64)
        self.D['pseudo_input']=tf.Variable(tf.ones([d,m,l],dtype=tf.float64),dtype=tf.float64)

    def get(self,name,l=None,n=None,d=None):
        if name=='pseudo_Sigma':
            if d!=None and n==None and l!=None:
                return tf.squeeze(tf.slice(self.D[name],[0,0,d-1,l-1],[-1,-1,1,1]))
        else:
            if d!=None and n==None and l!=None and name=='pseudo_mu':
                return tf.squeeze(tf.slice(self.D[name],[0,d-1,l-1],[-1,1,1]))
            if d==None and n==None and l!=None:
                return tf.squeeze(tf.slice(self.D[name],[0,0,l-1],[-1,-1,1]))
            elif d==None and l!=None and n!=None:
                return tf.squeeze(tf.slice(self.D[name],[0,n-1,l-1],[-1,1,1]))
            elif l!=None and d!=None and n!=None:
                return tf.squeeze(tf.slice(self.D[name],[0,n-1,l-1],[-1,1,1]))
        print('slicing not implemented:','(type,l,d,n)',name,l,d,n)



class Data:
    def __init__(self,n,d,X,Y):
        self.X=X
        self.Y=Y


