from math import sqrt
import numpy as np
from main.tensorOp import tensor_op
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
##################
# Initialization #
##################

def init_A_Sigma(y,N,T,P):
    """
    Use OLS method to initialize the tensor A
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A: N*N*d tensor
    """
    # create X (regressors)
    X = np.zeros((N*P,T-P))
    for i in range(P):
        X[i*N:i*N+N,:] = y[:,P-i-1:T-i-1]
    # create Y (response)
    Y = y[:,P:]
    # solve OLS
    A = (X@Y.T).T @ np.linalg.inv(X @ X.T)
    epsilon = Y - A@X
    Sigma = epsilon @ epsilon.T/(T-P) 
    # fold A into a tensor
    A = np.array(tensor_op.fold(A,(N,N,P),0))

    # # HOOI to get a low rank version
    # A,U = tucker(A,rank=[r1,r2,P])
    # A = tl.tenalg.multi_mode_dot(A,U)
    return A,Sigma

def spectral_init_A(y,N,T,P):
    """
    Spectral initialization
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N*P,T-P))
    for i in range(P):
        X[i*N:i*N+N,:] = y[:,P-i-1:T-i-1]
    # create Y (response)
    Y = y[:,P:]
    # spectral initialization
    A = np.zeros((N,N*P))
    for t in range(T-P):
        A = A + np.outer(Y[:,t],X[:,t])
    A = A/(T-P)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A,(N,N,P),0))

    # # HOOI to get a low rank version
    # A,U = tucker(A,rank=[r1,r2,P])
    # A = tl.tenalg.multi_mode_dot(A,U)
    return A

def rand_w(r,s): #checked
    """
    Uniform distribution for now
    (may need to adjust range for endpoint issue)
    """
    lmbd = np.random.rand(r) *2 -1 # (-1,1)
    gamma = np.random.rand(s) # (0,1)
    # phi = np.random.rand(s)*np.pi - np.pi/2 # (-pi/2,pi/2)
    phi = np.random.rand(s)*np.pi # (0,pi)
    return (lmbd,gamma,phi)

def get_L_MA(lmbd,eta,r,s,P): # checked
    """
    Compute the L_MA matrix given the parameters
    Set size to be P (truncated)
    """
    L = np.zeros((P,r+2*s))
    for i in range(P):
        for j in range(r):
            L[i,j] = np.power(lmbd[j],i+1)
        for j in range(s):
            L[i,r+2*j] = np.power(eta[0,j],i+1) * np.cos((i+1)*np.array(eta[1,j]))
            L[i,r+2*j+1] = np.power(eta[0,j],i+1) * np.sin((i+1)*np.array(eta[1,j]))
    # new = np.concatenate([L[:,:r], L[:,r:]],axis=1)
    return L


def get_L(lmbd,eta,r,s,P,p): # checked
    """
    Compute the L matrix given the parameters
    Set size to be P (truncated)
    """
    L_MA = get_L_MA(lmbd,eta,r,s,P-p)
    L = np.zeros((P,p+r+2*s))
    L[:p,:p] = np.identity(p)
    L[p:,p:] = L_MA
    return L # P*d matrix


def get_G(A,L):
    """
    Restore G from A and L
    G = A inv(L'L)L'
    """
    factor = np.matmul(np.linalg.pinv(np.matmul(L.T,L)),L.T) 
    G = mode_dot(A,factor,2)
    return G


##################
# Loss Functions #
##################

def loss_vec(Y,X1,G,L,T):
    '''
    2 * neg_loglike
    '''
    L = L[:T-1]

    epsilon = Y - (tensor_op.unfold(mode_dot(G,L,2),0).numpy() @ X1)
    loss = np.trace(epsilon @ epsilon.T)  
    
    return loss/(T-1)

###############
# Derivatives #
###############

"""
Prepare objective, Jacobian and Hessian functions
"""

def vec_jac_hess_lmbd(lmbd_k,k,G,L,Y,X1,X2,p,T): # checked
    """
    first(jac) and second(hess) derivative of L_ML w.r.t. lmbd_k
    """
    L_temp = np.copy(L[:(T-1),:])
    # L_temp[:,p+k] = 0
    # a = Y - (tensor_op.unfold(mode_dot(G,L_temp,2),0).numpy() @ X1) # N by (T-1)

    epsilon = Y - (tensor_op.unfold(mode_dot(G,L_temp,2),0).numpy() @ X1) # N by (T-1)
    power_series= np.arange(1,T-p)
    lmbd_power = np.power(lmbd_k,power_series)
    lmbd_y = np.einsum('i,jki->jki',lmbd_power,X2)
    # outer_grad = a - G[:,:,p+k] @ np.sum(lmbd_y,axis=2).T
    
    # first_grad: first derivative of eps w.r.t. lmbd_k 
    # second_grad: second derivative of eps w.r.t. lmbd_k 
    first_grad = -G[:,:,p+k] @ np.einsum('i,jki->jk',power_series,(lmbd_y/lmbd_k)).T
    second_grad = -G[:,:,p+k] @ np.einsum('i,i,jki->jk',power_series[:-1],power_series[1:],lmbd_y[:,:,1:]/(lmbd_k**2)).T
    
    summand_j = np.einsum('ij,ij->',first_grad, epsilon)
    summand_h = np.einsum('ij,ij->',second_grad, epsilon) + np.einsum('ij,ij->',first_grad, first_grad)
    return summand_j/(T-1),summand_h/(T-1)
# @jit(nopython=True, parallel=True)
def vec_jac_hess_gamma_phi(eta_k,k,G,L,Y,X1,X2,p,r,T):
    """ 
    first(jac) and second(hess) derivative of L_ML w.r.t. (gamma,phi) pair
    """
    gamma_k = eta_k[0]
    phi_k = eta_k[1]

    L_temp = np.copy(L[:(T-1),:])
    # L_temp[:,p+r+k:p+r+k+2] = 0 # set gamma_k and phi_k = 0 in L
    # a = Y - (tensor_op.unfold(mode_dot(G,L_temp,2),0).numpy() @ X1) 

    epsilon = Y - (tensor_op.unfold(mode_dot(G,L_temp,2),0).numpy() @ X1) 
    
    power_series = np.arange(1,T-p)
    gamma_power = np.power(gamma_k,power_series)
    
    cos_part = np.einsum('i,jki,i->jki',np.cos(phi_k*power_series), X2, gamma_power) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
    sin_part = np.einsum('i,jki,i->jki',np.sin(phi_k*power_series), X2, gamma_power) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
    cos_part_1 = np.einsum('i,jki->jki',power_series,(cos_part/gamma_k)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
    sin_part_1 = np.einsum('i,jki->jki',power_series,(sin_part/gamma_k)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
    cos_part_2 = np.einsum('i,jki->jki',power_series[:-1],cos_part_1[:,:,1:]/gamma_k) # (j-p) (j-p-1) gamma^(j-p-2) cos{(j-p)phi} y_{t-j}
    sin_part_2 = np.einsum('i,jki->jki',power_series[:-1],sin_part_1[:,:,1:]/gamma_k) # (j-p) (j-p-1) gamma^(j-p-2) sin{(j-p)phi} y_{t-j}
    A = G[:,:,p+r+2*k]   
    B = G[:,:,p+r+2*k+1]

    # first_grad_gamma: first derivative of eps w.r.t. gamma_k
    # first_grad_phi: first derivative of eps w.r.t. phi_k
    first_grad_gamma = -A @ np.sum(cos_part_1,axis=2).T - B @ np.sum(sin_part_1,axis=2).T
    first_grad_phi = A @ np.einsum('i,jki->jk',power_series,(sin_part)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part)).T
    
    # jac
    jac_gamma = np.einsum('ij,ij->',first_grad_gamma, epsilon)
    jac_phi = np.einsum('ij,ij->',first_grad_phi, epsilon)

    # hess
    hess_gg = np.einsum('ij,ij->',first_grad_gamma, first_grad_gamma) + np.einsum('ij,ij->', (-A @ np.sum(cos_part_2,axis=2).T - B @ np.sum(sin_part_2,axis=2).T),  epsilon)
    hess_gp = np.einsum('ij,ij->',first_grad_gamma, first_grad_phi) + np.einsum('ij,ij->', (A @ np.einsum('i,jki->jk',power_series,sin_part_1).T - B @ np.einsum('i,jki->jk',power_series,cos_part_1).T), epsilon)
    hess_pp = np.einsum('ij,ij->',first_grad_phi, first_grad_phi) + np.einsum('ij,ij->', (A @ np.einsum('i,jki->jk',power_series**2,cos_part).T + B @ np.einsum('i,jki->jk',power_series**2,sin_part).T), epsilon)
    return jac_gamma/(T-1),jac_phi/(T-1),np.array([[hess_gg,hess_gp],[hess_gp,hess_pp]])/(T-1)


##############
# Asymptotic #
##############

# Denote alpha'=(lmbd',eta')'

# Denote omega'=(lmbd',eta')'

# I = E(dl/dalpha*dl/dalpha')
# J = E(d^2/(dalpha*dalpha'))

def asymptotic_omega(eta,lmbd,G,Sigma,X2,p,r,s,T):
    '''
    return E(d^2L/(domega * domega)): (r+2s)*(r+2s) matrix
           E(dL/domega * dL/domega)): (r+2s)*(r+2s) matrix (equality)
    '''
    power_series= np.arange(1,T-p)

    I_omega = np.zeros((r+2*s,r+2*s))
    J_omega = np.zeros((r+2*s,r+2*s))
    for i in range(r+s):
        if i < r: # lmbd part
            lmbd_power_i = np.power(lmbd[i],power_series)
            lmbd_y_i = np.einsum('i,jki->jki',lmbd_power_i,X2)
            first_grad_lmbd_i = -G[:,:,p+i] @ np.einsum('i,jki->jk',power_series,(lmbd_y_i/lmbd[i])).T
            for j in range(i,r+s):
                if j < r : # lmbd part
                    lmbd_power_j = np.power(lmbd[j],power_series)
                    lmbd_y_j = np.einsum('i,jki->jki',lmbd_power_j,X2)
                    first_grad_lmbd_j = -G[:,:,p+j] @ np.einsum('i,jki->jk',power_series,(lmbd_y_j/lmbd[j])).T
                    I_omega[i,j] = np.einsum('ij,ij ->' , first_grad_lmbd_i, Sigma @ first_grad_lmbd_j )/(T-1)
                    J_omega[i,j] = np.einsum('ij,ij ->' , first_grad_lmbd_i,  first_grad_lmbd_j )/(T-1)
                else: # j>=r eta part
                    gamma_j, phi_j = eta[:,j-r]
                    gamma_power_j = np.power(gamma_j,power_series)
                    cos_part_j = np.einsum('i,jki,i->jki',np.cos(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
                    sin_part_j = np.einsum('i,jki,i->jki',np.sin(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
                    cos_part_1_j = np.einsum('i,jki->jki',power_series,(cos_part_j/gamma_j)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
                    sin_part_1_j = np.einsum('i,jki->jki',power_series,(sin_part_j/gamma_j)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
                    A = G[:,:,p+r+2*(j-r)]  
                    B = G[:,:,p+r+2*(j-r)+1]
                    first_grad_gamma_j = -A @ np.sum(cos_part_1_j,axis=2).T - B @ np.sum(sin_part_1_j,axis=2).T
                    first_grad_phi_j = A @ np.einsum('i,jki->jk',power_series,(sin_part_j)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_j)).T
                    I_omega[i,r+2*(j-r)] = np.einsum('ij,ij->',first_grad_lmbd_i ,Sigma @ first_grad_gamma_j )/(T-1)
                    I_omega[i,r+2*(j-r)+1] = np.einsum('ij,ij->',first_grad_lmbd_i ,Sigma @ first_grad_phi_j )/(T-1)
                    J_omega[i,r+2*(j-r)] = np.einsum('ij,ij->',first_grad_lmbd_i , first_grad_gamma_j )/(T-1)
                    J_omega[i,r+2*(j-r)+1] = np.einsum('ij,ij->',first_grad_lmbd_i , first_grad_phi_j )/(T-1)
        else: # i>=r eta part
            gamma_i, phi_i = eta[:,i-r]
            gamma_power_i = np.power(gamma_i,power_series)
            cos_part_i = np.einsum('i,jki,i->jki',np.cos(phi_i*power_series), X2, gamma_power_i) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
            sin_part_i = np.einsum('i,jki,i->jki',np.sin(phi_i*power_series), X2, gamma_power_i) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
            cos_part_1_i = np.einsum('i,jki->jki',power_series,(cos_part_i/gamma_i)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
            sin_part_1_i = np.einsum('i,jki->jki',power_series,(sin_part_i/gamma_i)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
            A = G[:,:,p+r+2*(i-r)]  
            B = G[:,:,p+r+2*(i-r)+1]
            first_grad_gamma_i = -A @ np.sum(cos_part_1_i,axis=2).T - B @ np.sum(sin_part_1_i,axis=2).T
            first_grad_phi_i = A @ np.einsum('i,jki->jk',power_series,(sin_part_i)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_i)).T
            for j in range(i,r+s):
                gamma_j, phi_j = eta[:,j-r]
                gamma_power_j = np.power(gamma_j,power_series)
                cos_part_j = np.einsum('i,jki,i->jki',np.cos(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
                sin_part_j = np.einsum('i,jki,i->jki',np.sin(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
                cos_part_1_j = np.einsum('i,jki->jki',power_series,(cos_part_j/gamma_j)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
                sin_part_1_j = np.einsum('i,jki->jki',power_series,(sin_part_j/gamma_j)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
                A = G[:,:,p+r+2*(j-r)]  
                B = G[:,:,p+r+2*(j-r)+1]
                first_grad_gamma_j = -A @ np.sum(cos_part_1_j,axis=2).T - B @ np.sum(sin_part_1_j,axis=2).T
                first_grad_phi_j = A @ np.einsum('i,jki->jk',power_series,(sin_part_j)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_j)).T
                if i == j:
                    # gg
                    I_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_gamma_j)/(T-1)
                    # gp
                    I_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_phi_j)/(T-1)
                    # pp
                    I_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i,Sigma @ first_grad_phi_j)/(T-1)
                    # gg
                    J_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_gamma_j)/(T-1)
                    # gp
                    J_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_phi_j)/(T-1)
                    # pp
                    J_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i, first_grad_phi_j)/(T-1)
                    
                else: # i =! j
                    # gg
                    I_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_gamma_j)/(T-1)
                    # gp
                    I_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_phi_j)/(T-1)
                    # pg
                    I_omega[2*(i-r)+r+1,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_phi_i,Sigma @ first_grad_gamma_j)/(T-1)
                    # pp
                    I_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i,Sigma @ first_grad_phi_j)/(T-1)
                    # gg
                    J_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_gamma_j)/(T-1)
                    # gp
                    J_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_phi_j)/(T-1)
                    # pg
                    J_omega[2*(i-r)+r+1,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_phi_i, first_grad_gamma_j)/(T-1)
                    # pp
                    J_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i, first_grad_phi_j)/(T-1)
                   
    I_omega = I_omega + I_omega.T - np.diag(np.diag(I_omega))
    J_omega = J_omega + J_omega.T - np.diag(np.diag(J_omega))
    return  I_omega,J_omega


def asymptotic_g(Sigma,z,N,T):
    '''
    return E(d^2L/(dg * dg)): N^2d*N^2d matrix
           E(dL/dg * dL/dg)): N^2d*N^2d matrix (equality)
    '''
    # for g
    # z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
    I_g = 0
    J_g = 0
    for t in range(T-1):
        first_grad = -np.kron(z[:,t], np.eye(N))
        I_g += first_grad.T @ Sigma @ first_grad/(T-1)
        J_g += first_grad.T @  first_grad/(T-1)
    return I_g,J_g



def asymptotic_omega_g(lmbd,eta,G,Sigma,z,X2,p,r,s,N,T):
    '''
    return E(d^2L/(domega * dg)): (r+2s)*(N^2 d) matrix
    '''
    power_series= np.arange(1,T-p)
    d = p+r+2*s
    # lmbd_g
    I_lmbd_g = np.zeros((r, N*N*d))
    J_lmbd_g = np.zeros((r, N*N*d))
    for i in range(r):
        lmbd_power_i = np.power(lmbd[i],power_series)
        lmbd_y_i = np.einsum('i,jki->jki',lmbd_power_i,X2)
        first_grad_lmbd_i = -G[:,:,p+i] @ np.einsum('i,jki->jk',power_series,(lmbd_y_i/lmbd[i])).T
        for t in range(T-1):
            first_grad_g = -np.kron(z[:,t], np.eye(N))
            I_lmbd_g[i] += first_grad_lmbd_i[:,t] @ Sigma @ first_grad_g/(T-1)
            J_lmbd_g[i] += first_grad_lmbd_i[:,t] @ first_grad_g/(T-1)
    # eta_g
    I_eta_g = np.zeros((2*s,N*N*d))
    J_eta_g = np.zeros((2*s,N*N*d))
    for j in range(s):
        gamma_j, phi_j = eta[:,j]
        gamma_power_j = np.power(gamma_j,power_series)
        cos_part_j = np.einsum('i,jki,i->jki',np.cos(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
        sin_part_j = np.einsum('i,jki,i->jki',np.sin(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
        cos_part_1_j = np.einsum('i,jki->jki',power_series,(cos_part_j/gamma_j)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
        sin_part_1_j = np.einsum('i,jki->jki',power_series,(sin_part_j/gamma_j)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
        A = G[:,:,p+r+2*j] 
        B = G[:,:,p+r+2*j+1]
        first_grad_gamma_j = -A @ np.sum(cos_part_1_j,axis=2).T - B @ np.sum(sin_part_1_j,axis=2).T
        first_grad_phi_j = A @ np.einsum('i,jki->jk',power_series,(sin_part_j)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_j)).T
        for t in range(T-1):
            first_grad_g = -np.kron(z[:,t], np.eye(N))
            I_eta_g[2*j] += first_grad_gamma_j[:,t] @ Sigma @ first_grad_g/(T-1)
            I_eta_g[2*j+1] += first_grad_phi_j[:,t] @ Sigma @ first_grad_g/(T-1)
            J_eta_g[2*j] += first_grad_gamma_j[:,t] @  first_grad_g/(T-1)
            J_eta_g[2*j+1] += first_grad_phi_j[:,t] @  first_grad_g/(T-1)
    return np.vstack((I_lmbd_g,I_eta_g)),np.vstack((J_lmbd_g,J_eta_g))




def asymptotic(lmbd,eta,G,Sigma,z,X2,epsilon,p,r,s,N,T):
    '''
    no. of all parameter: b=r+2*s+N^2*d+N(N-1)/2
    Return I: E(dL/dbeta * dL/dbeta') b*b matrix 
           J: E(dL^2/(dbeta*dbeta')) b*b matrix 
    '''
    d = p+r+2*s
    b = r+2*s+N*N*d # no. of all parameter
    # permutation matrix D (N^2 * N(N+1)/2): vec(Sigma) = D vech(Sigma)
    # D = np.zeros((N*N,N*(N+1)//2))
    # record = 0
    # for i in range(N):
    #     D[i*N+i:(i+1)*N, record : record + (N-i)]=np.eye(N-i)
    #     record_alt = i
    #     for j in range(i):
    #         D[i*N+j,record_alt]=1   
    #         record_alt  +=  (i-j)
    #     record =record +  (N-i)
        
    I = np.zeros((b,b))
    J = np.zeros((b,b))
    # omega, omega
    I[:r+2*s,:r+2*s]  = asymptotic_omega(eta,lmbd,G,Sigma,X2,p,r,s,T)[0]
    J[:r+2*s,:r+2*s]  = asymptotic_omega(eta,lmbd,G,Sigma,X2,p,r,s,T)[1]
    
    # g, g
    I[r+2*s:r+2*s+N*N*d,r+2*s:r+2*s+N*N*d] = asymptotic_g(Sigma,z,N,T)[0]
    J[r+2*s:r+2*s+N*N*d,r+2*s:r+2*s+N*N*d] = asymptotic_g(Sigma,z,N,T)[1]
    
    # # sigma, sigma
    # I[r+2*s+N*N*d:,r+2*s+N*N*d:] = asymptotic_sigma(Sigma_inv,epsilon,D,T)[0]
    # J[r+2*s+N*N*d:,r+2*s+N*N*d:] = asymptotic_sigma(Sigma_inv,epsilon,D,T)[1]

    # omega, g
    I[:r+2*s,r+2*s:r+2*s+N*N*d] = asymptotic_omega_g(lmbd,eta,G,Sigma,z,X2,p,r,s,N,T)[0]
    J[:r+2*s,r+2*s:r+2*s+N*N*d] = asymptotic_omega_g(lmbd,eta,G,Sigma,z,X2,p,r,s,N,T)[1]
    I[r+2*s:r+2*s+N*N*d,:r+2*s] = I[:r+2*s,r+2*s:r+2*s+N*N*d].T
    J[r+2*s:r+2*s+N*N*d,:r+2*s] = J[:r+2*s,r+2*s:r+2*s+N*N*d].T
    
    # # alpha, sigma
    # I[:r+2*s+N*N*d,r+2*s+N*N*d:] = asymptotic_alpha_sigma(lmbd,eta,G,Sigma_inv,z,X2,epsilon,D,p,r,s,N,T)
    # I[r+2*s+N*N*d:,:r+2*s+N*N*d] = I[:r+2*s+N*N*d,r+2*s+N*N*d:].T

    # return Gamma,Sigma
    return np.linalg.inv(J)@ I @ np.linalg.inv(J)/(T-1)

def DGP_VARMA(eps,Phi,Theta,N,T,burn):
    '''
    eps: N by T+burn
    '''
    y = np.zeros((N,T+burn))
    # eps = stats.multivariate_normal.rvs([0]*N,Sigma,T+burn)
    for t in range(T+burn):
        y[:,t] = Phi @ y[:,t-1] + eps[t] - Theta @ eps[t-1] 
    y=y[:,burn:]
    return y # N by T

# from scipy import stats
# from scipy.stats import ortho_group
# def DGP_VARMA(eps,eigenspace,N,T,burn,p,r,s):
#     # lmbd,gamma,theta 
#     # two coef matrices
#     if r == 1 and s == 0:
#         lmbd = [-0.7]
#         gamma = []
#         theta = []
#         # J = np.array([[-0.7]])
#         J = np.array([-0.7]) # 010 exp
#         H = np.array([0.5])
#     elif r == 1 and s == 1:
#         lmbd = [-0.7]
#         gamma = [0.8]
#         theta = [np.pi/4]
#         J = np.array([[-0.7,0,0],[0,0.7*np.sqrt(2)/2,0.7*np.sqrt(2)/2],[0,-0.7*np.sqrt(2)/2,0.7*np.sqrt(2)/2]])
#         H = np.array([[0.5,0,0],[0,-0.5,0],[0,0,-0.5]])
#         # H = np.array([[0.5,0,0,0],[0,-0.5,0,0],[0,0,-0.5,0],[0,0,0,0.5]])

#     elif r == 0 and s == 1:
#         lmbd = []
#         gamma = [0.8]
#         theta = [np.pi/4]
#         # J = np.array([[0.7*np.sqrt(2)/2,0.7*np.sqrt(2)/2],[-0.7*np.sqrt(2)/2,0.7*np.sqrt(2)/2]])
#         J = np.array([[0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2],[-0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2]]) # 001 exp
#         H = np.array([[-0.5,0],[0,-0.5]])
        
    
#     Theta = eigenspace[:,:r+2*s] @ J @ eigenspace[:,:r+2*s].T
#     # Theta1 = eigenspace[:,:r+2*s] @ np.array([[-0.5,0,0],[0,0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2],[0,-0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2]]) @ eigenspace[:,:r+2*s].T
#     Phi = eigenspace[:,:r+2*s] @ H @ eigenspace[:,:r+2*s].T
    
#     if p == 1:
#         B = eigenspace
#         B_minus = B.T #@ (Phi-Theta)
#         G = np.zeros((N,N,p+r+2*s))
#         # G[:,:,0] = eigenspace[:,:r+2*s] @ (H-J) @ eigenspace[:,:r+2*s].T
#         G[:,:,0] = Phi - Theta
#         for i in range(r):
#             G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:]) @ (Phi-Theta)
#         for i in range(s):
#             G[:,:,p+r+2*i] = (np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])) @ (Phi-Theta)
#             G[:,:,p+r+2*i+1] = (np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])) @ (Phi-Theta)
#         L = get_L(lmbd,gamma,theta,r,s,T,p)
#     elif p == 0:
#         Phi = np.zeros((N,N))
#         B = eigenspace
#         B_minus = -B.T
#         G = np.zeros((N,N,p+r+2*s))
#         for i in range(r):
#             G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])
#         for i in range(s):
#             G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])
#             G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])
#         L = get_L(lmbd,gamma,theta,r,s,T,p) 
#     y = np.zeros((N,T+burn))
#     # eps = stats.multivariate_normal.rvs([0]*N,Sigma,T+burn)
#     for t in range(T+burn):
#         y[:,t] = Phi @ y[:,t-1] - Theta @ eps[t-1] + eps[t]
#     y=y[:,burn:]
#     return y,G


# def get_sample_size_VARMADGP(N,p,r,s):
#     ratio = np.array([0.4,0.3,0.2,0.1])
#     r1 = r+2*s
#     d = r+2*s
#     NofP = N*r1*2 + r1*r1*(p+d)
#     ss = NofP/ratio
#     return np.array(np.round(ss),dtype=int)