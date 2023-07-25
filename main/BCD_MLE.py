"""
MLE for omega and G and Sigma
optimization method: Block coordinate descent
"""
import os
os.getcwd()
os.chdir('C:\\Users\\Administrator\\LYC\\SARMA\\SARMA_code')
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.tenalg import kronecker as kron # kron product
from tensorly.tenalg import mode_dot # tensor multiply

from main.help_function_for_MLE import * 
from main.tensorOp import * 



def BCD_MLE(y,p,r,s,P,n_iter=500,lmbd=None,gamma=None,phi=None,G=None,lr_omega=1,stop_thres=1e-4,stop_method='SepEst',result_show=False,flag_maxiter=0):
    '''
    Input:
    ------------------------------------
    y: data process ---N*T
    p: AR order
    r and s: MA order
    P: truncted order of VAR(P)
    n_iter: iterative number
    Paremeter initialization: lmbd (r), gamma (s), phi (s), G (N*N*d), Sigma (N*N)
    lr_omega: learning rate for omega
    stop_thres: 
    stop_method: 'SepEst': Seperate estimator (G,(lmbd,gamma,phi),Sigma) reach a stop limitation;
                 'Est'   : Coefficient A reach a stop limitation.
    flag_maxiter: records whether the iteration ended due to reaching n_iter.
    ------------------------------------
    Output:
    ------------------------------------
    A: N*N*T tensor
    lmbd: r array (-1,1)
    gamma: s array (0,1)
    phi: s array (-pi/2,pi/2)
    G: N*N*d tensor
    Loss: negative likelihood
    flag_maxiter
    ------------------------------------
    '''
    # Initialization
    N,T = y.shape
    d = p+r+2*s
    Loss = np.inf
    
    # If None, initial lmbd, gamma, phi
    if (lmbd is None) and (gamma is None) and (phi is None):
        lmbd,gamma,phi=rand_w(r,s)
        
    if G is None:
        eta = np.vstack((gamma,phi)) # 2*s matrix
        A, Sigma = init_A_Sigma(y,N,T,P) # A: N*N*P tensor, Sigma: N*N matrix
        # truncated L 
        L = get_L(lmbd,eta,r,s,P,p) # L: P*d matrix
        G = get_G(A,L) # G: N*N*d tensor
    eta = np.vstack((gamma,phi))

    L = get_L(lmbd,eta,r,s,T,p) # L: T*d matrix

    A = mode_dot(G,L,2) # A: N*N*T matrix


    
    # useful preparation for MLE
    Y = y[:,1:] # N*(T-1) matrix
    x = (np.flip(y,axis=1)).ravel(order='F') # NT array: vectorized from y_T to y_1
    X1 = np.zeros((N*(T-1),T-1)) # y_t = [A_1,...,A_{T-1}] * vec(y_{t-1},...,y_{t-T+1})
    for i in range(T-1):
        X1[:(i+1)*N,i] = x[-(i+1)*N:]
    
    X2 = np.zeros((T-1,N,T-p-1)) # shape: (T-1)*N*(T-p-1)  
    for i in range(p,T-1):
        X2[i,:,:(i+1-p)] = np.flip(y[:,:i+1-p],axis=1)
        
    # Initial sigma
    A = mode_dot(G,L,2) # N*N*T
    epsilon = Y-tensor_op.unfold(A[:,:,:T-1],0).numpy() @X1
    Sigma = (epsilon @ epsilon.T) / (T-1)
    Sigma_inv = np.linalg.inv(Sigma)
    
    power_series = np.arange(1,T-p+1)
    
    Loss_pre = loss_vec(Y,X1,G,L,Sigma,T)
    # BCD steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        # for lmbd iterative
        pre_lmbd = np.copy(lmbd)
        for k in range(r): 
            # update lmbd
            grad,hess =vec_jac_hess_lmbd(lmbd[k],k,G,L,Sigma_inv,Y,X1,X2,p,T)
            # grad1,hess1 = jac_hess_lmbd(lmbd[k],k,G,L,y,p,T)
            # lmbd[k] = lmbd[k] - lr_omega * jac_lmbd(lmbd[k],k,G,L,y,p,T) / hess_lmbd(lmbd[k],k,G,L,y,p,T)
            direct = grad/hess
            # 黄金分割法
            a = 0.5
            b = 2
            c = a+0.382*(b-a)
            d_ = a+0.618*(b-a)
            L_c = L.copy()
            L_d = L.copy()
            Loss_pre_gamma = loss_vec(Y,X1,G,L,Sigma,T)
            for j in range(10**(N)):
                lmbd_c = max(min(0.99,lmbd[k]- c*direct),-0.99)
                L_c[p:,p+k] = np.power(lmbd_c,power_series)
                fc = loss_vec(Y,X1,G,L_c,Sigma,T)
                
                lmbd_d = max(min(0.99,lmbd[k]- d_*direct),-0.99)
                L_d[p:,p+k] = np.power(lmbd_d,power_series)
                fd = loss_vec(Y,X1,G,L_d,Sigma,T)
                if fc < fd:
                    a = a
                    b = d_
                    d_ = c
                    c = a+0.382*(b-a)
                else:
                    b = b
                    a = c
                    c = d_
                    d_ = a+0.618*(b-a)
                if b-a < 0.1**(4):
                    stepsize = (a+b)/2
                    break
            # stepsize = (a+b)/2
            # print(stepsize)
            temp = lmbd[k] - stepsize * grad / hess
            lmbd[k] = max(min(0.99,temp),-0.99)
            L_c[p:,p+k] = np.power(lmbd[k],power_series)
            Loss_after_gamma = loss_vec(Y,X1,G,L_c,Sigma,T)
            if Loss_after_gamma> Loss_pre_gamma:
                continue
            L[p:,p+k] = np.power(lmbd[k],power_series)
            
            
            
            # temp = lmbd[k] - lr_omega * grad / hess
            # if temp > 0.9 or temp < -0.9:
            #     temp = lmbd[k] - 0.1*lr_omega * grad
            # lmbd[k] = max(min(0.9,temp),-0.9)
            # power_series = np.arange(1,T-p+1)
            # L[p:,p+k] = np.power(lmbd[k],power_series)
        # for eta=(gamma, phi) iterative
        pre_gamma = np.copy(gamma)
        pre_phi = np.copy(phi)
        for k in range(s):
            # update gamma and phi
            grad_gamma,grad_phi,hess = vec_jac_hess_gamma_phi([gamma[k],phi[k]],k,G,L,Sigma_inv,Y,X1,X2,p,r,T)
            # grad_gamma,grad_phi,hess = jac_hess_gamma_phi([gamma[k],phi[k]],k,G,L,y,p,r,T)
            grad = np.array([grad_gamma,grad_phi])
            hess_inv = np.linalg.inv(hess)
            
            direct = hess_inv@grad
            # 黄金分割法
            a = 0.5
            b = 2
            c = a+0.382*(b-a)
            d_ = a+0.618*(b-a)
            L_c = L.copy()
            L_d = L.copy()
            Loss_pre_eta = loss_vec(Y,X1,G,L,Sigma,T)
            for j in range(10**(N)):
                gamma_c = max(min(0.95,gamma[k] - c*direct[0]),0.05) 
                phi_c = max(min(0.95*np.pi,phi[k] - c*direct[1] ),0.05*np.pi)
                L_c[p:,p+r+2*k] = np.einsum('i,i->i',np.power(gamma_c,power_series),np.cos(power_series*phi_c))
                L_c[p:,p+r+2*k+1] = np.einsum('i,i->i',np.power(gamma_c,power_series),np.sin(power_series*phi_c))
                fc = loss_vec(Y,X1,G,L_c,Sigma,T)
                
                gamma_d = max(min(0.99,gamma[k] - d_*direct[0]),0.05) 
                phi_d = max(min(0.99*np.pi,phi[k] - d_*direct[1] ),0.05*np.pi)
                L_d[p:,p+r+2*k] = np.einsum('i,i->i',np.power(gamma_d,power_series),np.cos(power_series*phi_d))
                L_d[p:,p+r+2*k+1] = np.einsum('i,i->i',np.power(gamma_d,power_series),np.sin(power_series*phi_d))
                fd = loss_vec(Y,X1,G,L_d,Sigma,T)
                if fc < fd:
                    a = a
                    b = d_
                    d_ = c
                    c = a+0.382*(b-a)
                else:
                    b = b
                    a = c
                    c = d_
                    d_ = a+0.618*(b-a)
                if b-a < 0.1**(4):
                    stepsize = (a+b)/2
                    break
            # stepsize = (a+b)/2
            # print(stepsize)
            gamma[k] = max(min(0.99,gamma[k] - stepsize*direct[0]),0.05)
            phi[k] = max(min(0.99*np.pi,(phi[k] - stepsize*direct[1])),0.05*np.pi)
            L_c[p:,p+r+2*k] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.cos(power_series*phi[k]))
            L_c[p:,p+r+2*k+1] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.sin(power_series*phi[k]))
            Loss_after_eta = loss_vec(Y,X1,G,L_c,Sigma,T)
            
            if Loss_after_eta> Loss_pre_eta:
                continue      
            
            L[p:,p+r+2*k] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.cos(power_series*phi[k]))
            L[p:,p+r+2*k+1] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.sin(power_series*phi[k]))
        
            # temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            # if temp > 0.9 or temp < 0.1: 
            #     temp = gamma[k] - lr_omega * grad_gamma
            #     phi[k]= phi[k] - lr_omega * grad_phi
            # else:
            #     phi[k]= phi[k] - lr_omega * (hess_inv @ grad)[1]
            # gamma[k] = max(min(0.9,temp),0.1)
            # # phi[k] = max(min(np.pi/2,phi[k]),-np.pi/2) # phi \in (-pi/2,pi/2)
            # phi[k] = max(min(np.pi-0.0001,phi[k]),0.0001) # phi \in (0,pi)
            # power_series = np.arange(1,T-p+1)
            # L[p:,p+r+2*k] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.cos(power_series*phi[k]))
            # L[p:,p+r+2*k+1] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.sin(power_series*phi[k]))
        
        # update G
        z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
        # part_1 = 0
        # for t in range(T-1):
        #     part_1 += (Sigma_inv @ np.outer( Y[:,t],z[:,t])).ravel('F')
        G_1 =  ((Y@z.T) @ np.linalg.inv(z @ z.T))
        pre_G=G.copy()
        G = tensor_op.fold(G_1,(N,N,d),0).numpy()

        # update A
        pre_A = A 
        A = mode_dot(G,L,2) # N*N*T

        # update Sigma
        epsilon = Y-tensor_op.unfold(A[:,:,:T-1],0).numpy() @X1
        pre_Sigma = Sigma
        Sigma = (epsilon @ epsilon.T) / (T-1)
        Sigma_inv = np.linalg.inv(Sigma)
        
        Loss = loss_vec(Y,X1,G,L,Sigma,T)
        
        # diff
        omega_diff = 0
        if r>0 or s>0:
            omega_diff = np.linalg.norm(np.concatenate([lmbd-pre_lmbd,(gamma-pre_gamma).ravel('F'),(phi-pre_phi).ravel('F')]), ord=np.inf)
        G_diff = np.linalg.norm(tensor_op.unfold(G-pre_G,0), ord=np.inf)
        # omega_diff = np.linalg.norm(np.concatenate([lmbd-pre_lmbd,(gamma-pre_gamma).ravel('F'),(phi-pre_phi).ravel('F')]), ord=np.inf)
        Sigma_diff = np.linalg.norm(np.triu(Sigma-pre_Sigma),ord=np.inf)      
        A_diff = np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord=np.inf)
        Loss_diff = np.abs((Loss - Loss_pre) /Loss_pre)
        
        Loss_pre = Loss
        eta = np.vstack((gamma,phi))
        # early stop
        if (stop_method == 'SepEst') and \
            ((( G_diff < stop_thres) and (omega_diff < stop_thres) and (Sigma_diff < stop_thres)) or (Loss_diff < 0.01*stop_thres)):
            # Loss = loss_vec(Y,X1,G,L,Sigma,T)
            if result_show == True:
                print("=======================================\n",
                'Stop for reach SepEst limitation',
                f'Order: {p,r,s}; No. of iter: {iter_no} \n',
                f'omega_diff: {omega_diff}; G_diff: {G_diff}; Sigma_diff: {Sigma_diff}\n',
                f'Final Loss: {Loss}\n',
                f'Param: lmbd: {lmbd},\n gamma: {gamma},\n phi: {phi},\n G: {G}, \n Sigma: {Sigma}\n',
                "=============================================")
            result = pd.Series(dict(zip(['A', 'lmbd', 'eta','G','Sigma','Loss','flag_maxiter','iter_no'],
                     [A,lmbd,eta,G,Sigma,Loss,flag_maxiter,iter_no])))
            return result
        elif (stop_method == 'Est') and ((A_diff < stop_thres) or (Loss_diff < 0.01*stop_thres)):
            Loss = loss_vec(Y,X1,G,L,Sigma,T)
            if result_show == True:
                print("=======================================\n",
                'Stop for reach Est limitation',
                f'Order: {p,r,s}; No. of iter: {iter_no} \n',
                f'A_diff: {A_diff}',
                f'Final Loss: {Loss}\n',
                f'Param: lmbd: {lmbd},\n gamma: {gamma},\n phi: {phi},\n G: {G}, \n Sigma: {Sigma}\n',
                "=======================================")
            result = pd.Series(dict(zip(['A', 'lmbd', 'eta','G','Sigma','Loss','flag_maxiter','iter_no'],
                     [A,lmbd,eta,G,Sigma,Loss,flag_maxiter,iter_no])))
            return result
        
    flag_maxiter = 1
    Loss = loss_vec(Y,X1,G,L,Sigma,T)
    A = mode_dot(G,L,2)
    if result_show == True:
        print("=======================================\n",
        'Stop for reach iter_No limitation',
        f'Order: {p,r,s}; No. of iter: {iter_no} \n',
        f'omega_diff: {omega_diff}; G_diff: {G_diff}; Sigma_diff: {Sigma_diff}\n; A_diff: {A_diff}',
        f'Final Loss: {Loss}\n',
        f'Param: lmbd: {lmbd},\n gamma: {gamma},\n phi: {phi},\n G: {G}, \n Sigma: {Sigma}\n',
        "=============================================")
    result = pd.Series(dict(zip(['A', 'lmbd', 'eta','G','Sigma','Loss','flag_maxiter','iter_no'],
                     [A,lmbd,eta,G,Sigma,Loss,flag_maxiter,iter_no])))
    return result

if __name__ == "__main__" :
    
    
    # lmbd=np.array([-0.7]); gamma = np.array([0.7]); phi = np.array([np.pi/4])
    T=2000;N=3;p=0;r=1;s=1
    
    while True:
        # Generate a random NxN matrix
        A = np.random.uniform(low=-2, high=2, size=(N, N))
        
        # Use the Gram-Schmidt process to normalize each column
        for i in range(N):
            # A[:,i] = A[:,i]/np.linalg.norm(A[:,i])
            A[:,i] = A[:,i]
        
        # If the matrix is invertible (det != 0), then we're done
        if np.linalg.det(A) >0.1:
            break
    B=A
    
    # p=1;r=1;s=1
    
    burn=10000
    d=p+r+2*s
    P=int(np.floor(np.log(T)))
    n_iter = 50
    
    from scipy.stats import ortho_group
    B = ortho_group.rvs(N) 
    B_inv = np.linalg.inv(B)
    H = np.zeros((N,N))
    J = np.zeros((N,N))
    if p == 1:
        H = 0.5*np.eye(N)
    if p == 0:
        H = np.zeros((N,N))
    if r==1 and s==1:
        lmbd = [-0.8]; gamma = [0.8]; phi = [np.pi/4]
        J[0,0] = np.array(lmbd)
        J[1:3,1:3] = np.array([[0.8*np.cos(np.pi/4),0.8*np.sin(np.pi/4)],
                               [-0.8*np.sin(np.pi/4),0.8*np.cos(np.pi/4)]])
    if r==0 and s==1:
        lmbd = []; gamma = [0.8]; phi = [np.pi/4]
        J[0:2,0:2] = np.array([[0.8*np.cos(np.pi/4),0.8*np.sin(np.pi/4)],
                               [-0.8*np.sin(np.pi/4),0.8*np.cos(np.pi/4)]])
        # J = np.array([[0.8*np.cos(np.pi/4),0.8*np.sin(np.pi/4),0],[-0.8*np.sin(np.pi/4),0.8*np.cos(np.pi/4),0],[0,0,0]]) #r=0;s=1
    if r==1 and s==0:
        lmbd = [-0.8]; gamma = []; phi = []
        J[0,0] = np.array(lmbd)
        # J = np.array([[-0.8,0,0],[0,0,0],[0,0,0]]) #r=1;s=0
    if r==2 and s==0:
        lmbd = [-0.8,0.8]; gamma = []; phi = []
        J[0,0] = np.array(lmbd)[0]
        J[1,1] = np.array(lmbd)[1]
        # J = np.array([[-0.8,0,0],[0,0.8,0],[0,0,0]]) #r=2;s=0
    
    Phi = B @ H @ B_inv #AR
    Theta = B@ J @ B_inv #MA 

    G = np.zeros((N,N,p+r+2*s))
    
    if p == 1:
        G[:,:,0] = Phi - Theta
        # Gprime[:,:,0] = Phi - Theta
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_inv[i,:]) @ (Phi-Theta)
            # Gprime[:,:,p+i] = np.outer(B[:,i],B_minus[i,:]) @ (Phi-Theta)
        for i in range(s):
            G[:,:,p+r+2*i] = (np.outer(B[:,r+2*i],B_inv[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_inv[r+2*i+1,:])) @ (Phi-Theta)
            G[:,:,p+r+2*i+1] = (np.outer(B[:,r+2*i],B_inv[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_inv[r+2*i,:])) @ (Phi-Theta)

    elif p == 0:
        B_inv = -B_inv
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_inv[i,:])
        for i in range(s):
            G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_inv[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_inv[r+2*i+1,:])
            G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_inv[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_inv[r+2*i,:])
    
    # for k in range(p):
    #     G[:,:,k] = Phi - Theta
    # for i in range(r):
    #     G[:,:,i+p] = -np.outer(B[:,i],B_inv[i,:]) 
    # for j in range(s):
    #     G[:,:,p+r+2*j] = (np.outer(B[:,r+2*j],B_inv[r+2*j,:]) + np.outer(B[:,r+2*j+1],B_inv[r+2*j+1,:]))@ (Phi-Theta)
    #     G[:,:,p+r+2*j+1] = (np.outer(B[:,r+2*j],B_inv[r+2*j+1,:]) - np.outer(B[:,r+2*j+1],B_inv[r+2*j,:])) @ (Phi-Theta)

    #L = get_L(lmbd,gamma,phi,r,s,T,p)
    rep_n = 1
    y = np.zeros((N,T+burn))
    cov = 0.5*np.full((N,N),1)+0.5*np.eye(N)
    # cov = np.eye(N)
    eps = np.random.multivariate_normal(mean=np.zeros(N), cov =cov , size=T+burn).T
    for t in range(T+burn):
        y[:,t] = Phi @ y[:,t-1] + eps[:,t]- Theta @ eps[:,t-1] 
    y=y[:,burn:]
    # BCD_MLE(y,p,r,s,P,n_iter=200,result_show=True)
    a= BCD_MLE(y,p,r,s,P,n_iter,lmbd=np.array(lmbd), gamma = np.array([gamma]), phi = np.array([phi]), G=G,result_show=True)
    # a= BCD_MLE(y,p,r,s,P,n_iter,lmbd=np.array([-0.7]), gamma = np.array([0.7]), phi = np.array([np.pi/4]), G=G,result_show=True)

    np.linalg.norm(a['G']-G)
    (a['G']-G)[:,:,2]
    G[:,:,0]
    a['G'][:,:,0]
    tensor_op.unfold(a['G'],0).numpy()

