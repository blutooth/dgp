import maxParam as Param
import maxKern as kern
import maxKernStats as KernStats
import maxKL as KL
import tensorflow as tf
import numpy as np
import maxHelp as h
#import kernels as Kern

def Testing():
    m=5
    n=10
    d=3
    l=4

    M=Param.Model_Param(n,m,d,l)
    VM=Param.Var_Param(n,m,d,l)

    s=tf.Session()
    s.run(tf.initialize_all_variables())



    kernel=kern.RBF(d)


    # Computing KL_Divergences between Terms
    H_mu=VM.get('point_mu',l=1)
    H_mu2=VM.get('point_mu',l=1,n=2)
    H_std=VM.get('point_Sigma',l=1)
    H_std2=VM.get('point_Sigma',l=1,n=2)
    H_std1=VM.get('point_Sigma',l=1,n=1)
    U_mu=VM.get('pseudo_mu',l=1,d=1)
    U_mu2=VM.get('pseudo_mu',l=1,d=2)
    ps_in=VM.get('pseudo_input',l=1)
    H_mu1=VM.get('point_mu',l=1,n=1)
    U_var=VM.get('pseudo_Sigma',l=1,d=1)
    U_var2=VM.get('pseudo_Sigma',l=1,d=2)


    print('test-kern',s.run(kernel.K(tf.transpose(H_mu))))

    KL.setSession(s)
    h.set_sess(s)
    KernStats.getSession(s)


    #print(s.run(H_mu),s.run(H_std),s.run(H_mu1),s.run(H_std1),n)

    #KL_div=KL.KL_diag(H_mu1,H_std1,H_mu2,H_std2,d)
    KL_div2=KL.KL_full(U_mu,U_var,U_mu2,U_var2,d)
    #print('u_var2')
    #print(s.run(U_var2))
    #print(s.run(U_var))
    #l=KernStats.build_psi_stats_rbf(Z, kern, mu, S)
    l=KernStats.build_psi_stats_rbf(ps_in,M.D['Kernel',1,1],H_mu,H_std)
    print('testl',s.run(l))

    print(s.run(KL_div2))
    print(1)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here

   Testing()


