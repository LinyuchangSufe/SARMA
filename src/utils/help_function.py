import numpy as np
# import tensorly as tl
# from tensorly.decomposition import tucker
# from tensorly.tenalg import kronecker as kron
# from tensorly.tenalg import mode_dot
# from Application.current_analysis import X2
from src.utils.tensorOp import tensor_op
##################
# Initialization #
##################

def get_A(lmbd, eta, G, p,r,s, P):
    """
    lmbd: (r,) array
    eta: (s,2) array, each row is (gamma, phi)
    G: (N, N, p+r+2s) tensor
    return A: (N,N,10) matrix
    """
    r = np.size(lmbd)
    s = np.size(eta)//2
    N, _, _ = G.shape
    A = np.zeros((N,N,P))
    A[:,:,:p] = G[:,:,:p]  
    for k in range(p, P):
        for i in range(r):
            A[:,:,k] += lmbd[i]**(k-p+1) * G[:,:,p+i]
        for i in range(s):
            A[:,:,k] += eta[i,0]**(k-p+1) * (np.cos((k-p+1) * eta[i,1]) * G[:,:,p+r+2*i] +
                                             np.sin((k-p+1) * eta[i,1]) * G[:,:,p+r+2*i+1])
    return A

def gen_X_AR(y,p):
    T, N = y.shape
    X_AR = np.zeros((T-p,N,p))
    for i in range(p):
        X_AR[:,:,i] = y[p-1-i:T-1-i]
    return X_AR

def update_X_lmbd(X_lmbd,lmbd_k,k,y,p,P):
    T, N = y.shape
    P = min(T-p, P)
    lmbd_power = np.power(lmbd_k, np.arange(1, P)) # truncated
    for j in range(N):
        X_lmbd[1:, j, k] = np.convolve(lmbd_power, y[:T-p-1,j], mode= 'full')[:T-p-1] # sum lmbd^k y_{t-k-p}
    return X_lmbd

def gen_X_lmbd(lmbd,y,p,P):
    T, N = y.shape
    P = min(T-p, P)
    lmbd = np.array(lmbd)
    r = np.size(lmbd)//1
    X_lmbd = np.zeros((T-p, N, r))
    for k in range(r):
        X_lmbd = update_X_lmbd(X_lmbd,lmbd[k],k,y,p,P) # sum lmbd^k y_{t-k-p}
    return X_lmbd


def update_X_eta(X_eta,eta_k,k,y,p,P):
    T, N = y.shape
    P = min(T-p, P)
    
    indices = np.arange(1, P)
    cos_series = np.power(eta_k[0], indices) * np.cos(indices * eta_k[1]) 
    sin_series = np.power(eta_k[0], indices) * np.sin(indices * eta_k[1]) 
    for j in range(N):
        X_eta[1:, j, 2*k] = np.convolve(cos_series, y[:T-p-1,j], mode= 'full')[:T-p-1]
        X_eta[1:, j, 2*k+1] = np.convolve(sin_series, y[:T-p-1,j], mode= 'full')[:T-p-1]
    return X_eta

def gen_X_eta(eta,y,p,P):
    T, N = y.shape
    P = min(T-p, P)
    eta = np.reshape(eta,(-1,2))
    s = np.size(eta)//2
    X_eta = np.zeros((T-p, N, 2 * s))
    for k in range(s):
        X_eta = update_X_eta(X_eta,eta[k],k,y,p,P) # sum gamma^k cos{(k-p)phi} y_{t-k-p}, sum gamma^k sin{(k-p)phi} y_{t-k-p}
    return X_eta

def get_y_pre(lmbd, eta,G, y, p,r,s,P):
    y = np.vstack((y,np.zeros((1,y.shape[1]))))  # pad zeros for convolution
    y_pre = 0
    X_AR = gen_X_AR(y,p)
    X_lmbd = gen_X_lmbd(lmbd,y,p,P)
    X_eta = gen_X_eta(eta,y,p,P)
    for i in range(p):
        y_pre += X_AR[:,:,i] @ G[:,:,i].T
    for i in range(r):
        y_pre += X_lmbd[:,:,i] @ G[:,:,p+i].T
    for i in range(s):
        y_pre += X_eta[:,:,2*i] @ G[:,:,p+r+2*i].T + X_eta[:,:,2*i+1] @ G[:,:,p+r+2*i+1].T 
    return y_pre # (T-p) by N

##################
# Loss Functions #
##################

def get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s):
    epsilon = y[p:].copy()
    for i in range(p):
        epsilon -= X_AR[:,:,i] @ G[:,:,i].T
    for i in range(r):
        epsilon -= X_lmbd[:,:,i] @ G[:,:,p+i].T
    for i in range(s):
        epsilon -= X_eta[:,:,2*i] @ G[:,:,p+r+2*i].T + X_eta[:,:,2*i+1] @ G[:,:,p+r+2*i+1].T 
    return epsilon # (T-p) by N

def loss_gls(G, Sigma_inv,y, X_AR, X_lmbd, X_eta, p, r, s):
    '''
    2 * neg_loglike
    '''
    T, _ = y.shape
    epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
    loss = np.trace(epsilon.T @ epsilon @ Sigma_inv)
    return loss/(T-p)


def loss_mle(G, Sigma_inv,y, X_AR, X_lmbd, X_eta, p, r, s):
    '''
    2 * neg_loglike
    '''
    T, _ = y.shape
    epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s)
    loss = -(T-p) * np.log(np.linalg.det(Sigma_inv)) + np.trace(epsilon.T @ epsilon @ Sigma_inv)
    return loss/(T-p)


###############
# Derivatives #
###############

"""
Prepare objective, Jacobian and Hessian functions
"""

def vec_jac_lmbd(lmbd_k,k, G, Sigma_inv, y,X_AR,X_lmbd,X_eta, p,r,s,P): 
    """
    first(jac) derivative of L_ML w.r.t. lmbd_k
    """
    epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s) # (T-p) by N
    T, N = y.shape
    P = min(T-p, P)
    X_lmbd_k = np.zeros((T-p, N))
    indices = np.arange(1, P)
    lmbd_power = np.power(lmbd_k, indices-1)
    for j in range(N):
        X_lmbd_k[1:, j] = np.convolve(indices * lmbd_power, y[:T-p-1,j], mode= 'full')[:T-p-1] # h lmbd^{h-1} y_{t-p-1}

    first_grad = - X_lmbd_k @ G[:,:,p+k].T
    summand_jac = np.trace(epsilon.T @ first_grad @ Sigma_inv)/(T-p)
    # summand_jac = np.mean(np.sum(first_grad * epsilon, axis=1))
    return np.array([summand_jac])

def vec_jac_hess_lmbd(lmbd_k,k,G, Sigma_inv,y,X_AR,X_lmbd,X_eta, p,r,s,P): # checked
    """
    first(jac) and second(hess) derivative of L_ML w.r.t. lmbd_k
    """
    epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s) # (T-p) by N
    T, N = y.shape
    P = min(T-p, P)
    X_lmbd_k_1 = np.zeros((T-p, N))
    X_lmbd_k_2 = np.zeros((T-p, N))
    indices = np.arange(1, P)
    lmbd_power = np.power(lmbd_k, indices-1)
    for j in range(N):
        X_lmbd_k_1[1:, j] = np.convolve(indices * lmbd_power, y[:T-p-1,j], mode= 'full')[:T-p-1] # h lmbd^{h-1} y_{t-p-1}
        X_lmbd_k_2[2:, j] = np.convolve((indices + 1) * indices * lmbd_power, y[:T-p-2,j], mode= 'full')[:T-p-2]
    first_grad = - X_lmbd_k_1 @ G[:,:,p+k].T
    second_grad = - X_lmbd_k_2 @ G[:,:,p+k].T
    summand_jac = np.trace(epsilon.T @ first_grad @ Sigma_inv)/(T-p)
    summand_hess = np.trace(epsilon.T @ second_grad @ Sigma_inv)/(T-p) + np.trace(first_grad.T @ first_grad @ Sigma_inv)/(T-p)
    return np.array([summand_jac]), np.array([summand_hess])

    # summand_jac = np.mean(np.sum(first_grad * epsilon, axis=1))
    # summand_hess = np.mean(np.sum(second_grad * epsilon, axis=1)) + np.mean(np.sum(first_grad * first_grad, axis=1))
    # return summand_jac, summand_hess

# @jit(nopython=True, parallel=True)
def vec_jac_eta(eta_k,k,G, Sigma_inv,y,X_AR,X_lmbd,X_eta,p,r,s,P):
    """ 
    first(jac) derivative of L_ML w.r.t. eta(gamma,phi) pair
    """
    epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s) # (T-p) by N
    T, N = y.shape
    P = min(T-p, P)
    X_eta_k = np.zeros((T-p, N,2))
    indices = np.arange(1, P)
    cos_series = np.power(eta_k[0], indices-1) * np.cos(indices * eta_k[1]) 
    sin_series = np.power(eta_k[0], indices-1) * np.sin(indices * eta_k[1]) 
    for j in range(N):
        X_eta_k[1:,j,0] = np.convolve(indices * cos_series, y[:T-p-1,j], mode= 'full')[:T-p-1]
        X_eta_k[1:,j,1] = np.convolve(indices * sin_series, y[:T-p-1,j], mode= 'full')[:T-p-1]

    first_grad_gamma = - X_eta_k[:,:,0] @ G[:,:,p+r+2*k].T - X_eta_k[:,:,1] @ G[:,:,p+r+2*k+1].T
    first_grad_phi = eta_k[0] * X_eta_k[:,:,1] @ G[:,:,p+r+2*k].T - eta_k[0] * X_eta_k[:,:,0] @ G[:,:,p+r+2*k+1].T
    
    # jac
    jac_gamma = np.trace(epsilon.T @ first_grad_gamma @ Sigma_inv)/(T-p)
    jac_phi = np.trace(epsilon.T @ first_grad_phi @ Sigma_inv)/(T-p)

    return np.array([jac_gamma,jac_phi]) 

def vec_jac_hess_eta(eta_k,k,G, Sigma_inv,y,X_AR,X_lmbd,X_eta,p,r,s,P):
    """ 
    first(jac) and second(hess) derivative of L_ML w.r.t. eta(gamma,phi) pair
    """
    epsilon = get_epsilon(G, y, X_AR, X_lmbd, X_eta, p, r, s) # (T-p) by N
    T, N = y.shape
    P = min(T-p, P)
    X_eta_k_1 = np.zeros((T-p, N,2))
    X_eta_k_2 = np.zeros((T-p, N,2,2))
    indices = np.arange(1, P)
    power_series = np.power(eta_k[0], indices-1)
    cos_series = np.cos(indices * eta_k[1]) 
    sin_series = np.sin(indices * eta_k[1]) 
    for j in range(N):
        X_eta_k_1[1:,j,0] = np.convolve(indices * power_series * cos_series, y[:T-p-1,j], mode= 'full')[:T-p-1]
        X_eta_k_1[1:,j,1] = np.convolve(indices * power_series * sin_series, y[:T-p-1,j], mode= 'full')[:T-p-1]
        X_eta_k_2[2:,j,0,0] = np.convolve((indices + 1)[:-1] * indices[:-1] * power_series[:-1] * cos_series[1:], y[:T-p-2,j], mode= 'full')[:T-p-2] # h(h-1)gamma^{h-2}cos y_{t-h}
        X_eta_k_2[2:,j,1,0] = np.convolve((indices + 1)[:-1] * indices[:-1] * power_series[:-1] * sin_series[1:], y[:T-p-2,j], mode= 'full')[:T-p-2] # h(h-1)gamma^{h-2}sin y_{t-h}
        X_eta_k_2[1:,j,0,1] = np.convolve(indices * indices * power_series * cos_series, y[:T-p-1,j], mode= 'full')[:T-p-1] # h^2gamma^{h-1}cos y_{t-h}
        X_eta_k_2[1:,j,1,1] = np.convolve(indices * indices * power_series * sin_series, y[:T-p-1,j], mode= 'full')[:T-p-1] # h^2gamma^{h-1}sin y_{t-h}
    first_grad_gamma = - X_eta_k_1[:,:,0] @ G[:,:,p+r+2*k].T - X_eta_k_1[:,:,1] @ G[:,:,p+r+2*k+1].T
    first_grad_phi = eta_k[0] * X_eta_k_1[:,:,1] @ G[:,:,p+r+2*k].T - eta_k[0] * X_eta_k_1[:,:,0] @ G[:,:,p+r+2*k+1].T
    
    # jac
    jac_gamma = np.trace(epsilon.T @ first_grad_gamma @ Sigma_inv)/(T-p)
    jac_phi = np.trace(epsilon.T @ first_grad_phi @ Sigma_inv)/(T-p)


    # hess
    hess_gg = np.trace(first_grad_gamma.T @ first_grad_gamma @ Sigma_inv)/(T-p)+ \
        np.mean(np.sum((-X_eta_k_2[:,:,0,0] @ G[:,:,p+r+2*k].T  - X_eta_k_2[:,:,1,0] @ G[:,:,p+r+2*k+1].T ) * (epsilon @ Sigma_inv), axis = 1))
       
    hess_gp = np.trace(first_grad_phi.T @ first_grad_gamma @ Sigma_inv)/(T-p) + \
        np.mean(np.sum((X_eta_k_2[:,:,1,1] @ G[:,:,p+r+2*k].T  - X_eta_k_2[:,:,0,1] @ G[:,:,p+r+2*k+1].T ) * (epsilon @ Sigma_inv), axis = 1))
        
    hess_pp = np.trace(first_grad_phi.T @ first_grad_phi @ Sigma_inv)/(T-p) + \
        np.mean(np.sum(eta_k[0] * (X_eta_k_2[:,:,0,1] @ G[:,:,p+r+2*k].T  + X_eta_k_2[:,:,1,1] @ G[:,:,p+r+2*k+1].T ) * (epsilon @ Sigma_inv), axis = 1))

    
    return np.array([jac_gamma,jac_phi]), np.array([[hess_gg,hess_gp],[hess_gp,hess_pp]])
###############
#  Optimizer  #
###############


from scipy.optimize import minimize, Bounds

def optimize_parameter(
    init_val,                   # 初始值 (标量或 array)
    update_design_matrix_fn,    # 更新设计矩阵函数 (lam or eta)
    vec_jac_fn,                 # 返回 grad 的函数
    vec_jac_hess_fn,            # 返回 grad 和 hess 的函数
    loss_fn,                    # 计算 loss 的函数
    bounds,                     # 参数边界 [(low1, high1), (low2, high2), ...]
    lr=1e-1, warmup=100, n_steps=500, beta1=0.9, beta2=0.99,
    tol=1e-6, eps=1e-8, reg=1e-6, cache_eps=1e-5,
    warmup_method='lbfgs', # 新增：'adam' or 'lbfgs'
    refine_method=True  
):
    """
    通用的连续参数优化器，支持 Adam or L-BFSG-B warm-up + trust-region Newton refinement。
    """

    param = np.array(init_val, dtype=float)
    
    def fun(x):
        update_cache(x)
        return cache['loss']

    def jac(x):
        update_cache(x)
        return cache['grad']

    def hess(x):
        update_cache(x)
        return cache['hess']

    # ---------------- warm-up ----------------
    if warmup_method == 'adam':
        m = np.zeros_like(param)
        v = np.zeros_like(param)
        for t in range(1, warmup + 1):
            grad = vec_jac_fn(param)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            lr_t = lr * 0.9 ** (t // 5)
            step = lr_t * m_hat / (np.sqrt(v_hat) + eps)
            lower_bounds, upper_bounds = zip(*bounds)
            param_new = np.clip(param - step, lower_bounds, upper_bounds)
            update_design_matrix_fn(param_new)
            if np.max(np.abs(step)) < tol * 10:
                break
            param = param_new
    elif warmup_method == 'lbfgs':
        def update_cache(param_val):
            param_arr = np.array(param_val, dtype=float)
            if np.max(np.abs(param_arr - cache['val'])) > cache_eps:
                cache['val'] = param_arr
                update_design_matrix_fn(param_arr)
                cache['loss'] = loss_fn()
                cache['grad'] = vec_jac_fn(param_arr)
        cache = {'val': np.full_like(param, np.inf),
                 'loss': None, 'grad': None, 'hess': None}
        warmup_result = minimize(
            fun,
            x0=param,
            jac=jac,
            bounds=Bounds(*zip(*bounds)),
            method='L-BFGS-B',
            options={"gtol": tol , "maxiter": warmup}
        )
        param = warmup_result.x
        update_design_matrix_fn(param)

    else:
        raise ValueError(f"Unknown warmup_method: {warmup_method}")

    # ---------------- trust-region with cache ----------------
    if refine_method:
        def update_cache(param_val):
            param_arr = np.array(param_val, dtype=float)
            if np.max(np.abs(param_arr - cache['val'])) > cache_eps:
                cache['val'] = param_arr
                update_design_matrix_fn(param_arr)
                cache['loss'] = loss_fn()
                cache['grad'], h = vec_jac_hess_fn(param_arr)
                cache['hess'] = reg * np.diag(np.ones_like(cache['grad'])) + h

        cache = {'val': np.full_like(param, np.inf),
                'loss': None, 'grad': None, 'hess': None}
        res = minimize(fun, x0=param, jac=jac, hess=hess,
                    method="trust-constr",
                    bounds=Bounds(*zip(*bounds)),
                    options={"gtol": tol, "maxiter": n_steps - warmup})
        param = res.x
        update_design_matrix_fn(param)
    return param


def get_G(Y, Z):
    """
    Y: （T-p）* N
    Z: (T-p) * N * d
    return G: (N, N, d) tensor
    """
    T_, N = Y.shape
    _, _, d = Z.shape
    G = np.empty((d, N))
    Z_vec = Z.transpose(0, 2, 1).reshape(T_, -1)  # (T-p) * Nd
    A = Z_vec.T @ Z_vec  # shape (Nd, Nd)
    B = Y.T @ Z_vec  # shape (N, Nd)
    try:
        # Try fast solve
        G_flat = np.linalg.solve(A.T, B.T).T  # shape (N, Nd)
    except np.linalg.LinAlgError:
        # Fallback to least-squares
        G_flat = np.linalg.lstsq(A.T, B.T, rcond=None)[0].T  # still (N, Nd)

    # Reshape to (N, N, d) — fold N*d axis
    return tensor_op.fold(G_flat, (N, N, d), 0).numpy()
##############
# Asymptotic #
##############

# Denote alpha'=(lmbd',eta')'

# Denote omega'=(lmbd',eta')'

# I = E(dl/dalpha*dl/dalpha')
# J = E(d^2/(dalpha*dalpha'))


def asymptotic_omega_g_sigma(lmbd,eta,G,Sigma,y,epsilon,X_AR,X_lmbd,X_eta,p,r,s,P,method,D):
    '''
    return J = E(d^2L/(domega * domega)): (r+2s)*(r+2s) matrix
           I = E(dL/domega * dL/domega)): (r+2s)*(r+2s) matrix (equality)
    '''
    T, N = y.shape
    X = np.concatenate((X_AR, X_lmbd, X_eta), axis = 2)

    Sigma_inv = np.linalg.inv(Sigma)
    part_S2 = 0 # E(epsilon * vec(epsilon*epsilon')')
    for t in range(T-1):
        part_S2 += np.outer(epsilon[t], np.outer(epsilon[t],epsilon[t]).ravel('F'))/(T-1) 

    X_lmbd_k = np.zeros((T-p, N))
    first_grad_lmbd = np.zeros((r, T-p, N))
    first_grad_phi = np.zeros((s, T-p, N))
    first_grad_gamma = np.zeros((s, T-p, N))
    
    I_omega = np.zeros((r+2*s,r+2*s))
    J_omega = np.zeros((r+2*s,r+2*s))
    d = p+r+2*s

    # lmbd_g
    I_lmbd_g = np.zeros((r, N*N*d))
    J_lmbd_g = np.zeros((r, N*N*d))
    # lmbd_sigma
    I_lmbd_sigma = np.zeros((r,N*(N+1)//2))

    # eta_g
    I_eta_g = np.zeros((2*s,N*N*d))
    J_eta_g = np.zeros((2*s,N*N*d))

    # eta_sigma
    I_eta_sigma = np.zeros((2*s,N*(N+1)//2))
    
    # g_sigma
    I_g_sigma = 0

    indices = np.arange(1, P)

    # lmbd
    for i in range(r):
        lmbd_power_i = np.power(lmbd[i], indices-1)
        for k in range(N):
            X_lmbd_k[1:, k] = np.convolve(indices * lmbd_power_i, y[:T-p-1,k], mode= 'full')[:T-p-1] # h lmbd^{h-1} y_{t-p-1}
        first_grad_lmbd[i] = - X_lmbd_k @ G[:,:,p+i].T
    # eta
    for j in range(s):
        cos_series = np.power(eta[j][0], indices-1) * np.cos(indices * eta[j][1]) 
        sin_series = np.power(eta[j][0], indices-1) * np.sin(indices * eta[j][1]) 
        X_eta_k = np.zeros((T-p, N,2))
        for k in range(N):
            X_eta_k[1:,k,0] = np.convolve(indices * cos_series, y[:T-p-1,k], mode= 'full')[:T-p-1]
            X_eta_k[1:,k,1] = np.convolve(indices * sin_series, y[:T-p-1,k], mode= 'full')[:T-p-1]

        first_grad_gamma[j] = - X_eta_k[:,:,0] @ G[:,:,p+r+2*j].T - X_eta_k[:,:,1] @ G[:,:,p+r+2*j+1].T
        first_grad_phi[j] = eta[j][0] * X_eta_k[:,:,1] @ G[:,:,p+r+2*j].T - eta[j][0] * X_eta_k[:,:,0] @ G[:,:,p+r+2*j+1].T
        

    # lmbd - lmbd (or eta)
    for i in range(r):
        for j in range(i,r):
            if method == 'ls':
                I_omega[i,j] = np.einsum('ij,ij ->' , first_grad_lmbd[i], first_grad_lmbd[j] @ Sigma )/(T-1)
                J_omega[i,j] = np.einsum('ij,ij ->' , first_grad_lmbd[i], first_grad_lmbd[j] )/(T-1)
            else:
                I_omega[i,j] = np.einsum('ij,ij ->' , first_grad_lmbd[i], first_grad_lmbd[j] @ Sigma_inv )/(T-1)
                J_omega[i,j] = I_omega[i,j]
    for i in range(r):
        for j in range(s):
            if method == 'ls':
                I_omega[i,r+2*(j)] = np.einsum('ij,ij->',first_grad_lmbd[i] , first_grad_gamma[j] @ Sigma )/(T-1)
                I_omega[i,r+2*(j)+1] = np.einsum('ij,ij->',first_grad_lmbd[i], first_grad_phi [j] @ Sigma )/(T-1)
                J_omega[i,r+2*(j)] = np.einsum('ij,ij->',first_grad_lmbd[i] , first_grad_gamma[j] )/(T-1)
                J_omega[i,r+2*(j)+1] = np.einsum('ij,ij->',first_grad_lmbd[i] , first_grad_phi[j] )/(T-1)
            else:
                I_omega[i,r+2*(j)] = np.einsum('ij,ij->',first_grad_lmbd[i] , first_grad_gamma[j] @ Sigma_inv )/(T-1)
                I_omega[i,r+2*(j)+1] = np.einsum('ij,ij->',first_grad_lmbd[i], first_grad_phi[j] @ Sigma_inv )/(T-1)
                J_omega[i,r+2*(j)] = I_omega[i,r+2*(j)]
                J_omega[i,r+2*(j)+1] = I_omega[i,r+2*(j)+1]

    # eta - eta
    for i in range(s):
        for j in range(i, s):
            if i == j and method == 'ls':
                # gg
                I_omega[2*(i)+r,2*(j)+r] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_gamma[j] @ Sigma)/(T-1)
                # gp
                I_omega[2*(i)+r,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_phi[j] @ Sigma)/(T-1)
                # pp
                I_omega[2*(i)+r+1,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_phi[j] @ Sigma)/(T-1)
                # gg
                J_omega[2*(i)+r,2*(j)+r] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_gamma[j])/(T-1)
                # gp
                J_omega[2*(i)+r,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_phi[j])/(T-1)
                # pp
                J_omega[2*(i)+r+1,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_phi[j])/(T-1)
            elif i==j and method != 'ls':
                # gg
                I_omega[2*(i)+r,2*(j)+r] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_gamma[j] @ Sigma_inv)/(T-1)
                # gp
                I_omega[2*(i)+r,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_phi[j] @ Sigma_inv)/(T-1)
                # pp
                I_omega[2*(i)+r+1,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_phi[j] @ Sigma_inv)/(T-1)
                # gg
                J_omega[2*(i)+r,2*(j)+r] = I_omega[2*(i)+r,2*(j)+r]
                # gp
                J_omega[2*(i)+r,2*(j)+r+1] = I_omega[2*(i)+r,2*(j)+r+1]
                # pp
                J_omega[2*(i)+r+1,2*(j)+r+1] = I_omega[2*(i)+r+1,2*(j)+r+1]
                
            elif i!=j and method =='ls': # i =! j
                # gg
                I_omega[2*(i)+r,2*(j)+r] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_gamma[j] @ Sigma)/(T-1)
                # gp
                I_omega[2*(i)+r,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_phi[j] @ Sigma)/(T-1)
                # pg
                I_omega[2*(i)+r+1,2*(j)+r] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_gamma[j] @ Sigma)/(T-1)
                # pp
                I_omega[2*(i)+r+1,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_phi[j] @ Sigma)/(T-1)
                # gg
                J_omega[2*(i)+r,2*(j)+r] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_gamma[j])/(T-1)
                # gp
                J_omega[2*(i)+r,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_phi[j])/(T-1)
                # pg
                J_omega[2*(i)+r+1,2*(j)+r] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_gamma[j])/(T-1)
                # pp
                J_omega[2*(i)+r+1,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_phi[j])/(T-1)
            elif i!=j and method !='ls': # i =! j
                # gg
                I_omega[2*(i)+r,2*(j)+r] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_gamma[j] @ Sigma_inv)/(T-1)
                # gp
                I_omega[2*(i)+r,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_gamma[i], first_grad_phi[j] @ Sigma_inv)/(T-1)
                # pg
                I_omega[2*(i)+r+1,2*(j)+r] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_gamma[j] @ Sigma_inv)/(T-1)
                # pp
                I_omega[2*(i)+r+1,2*(j)+r+1] = np.einsum('ij,ij->',first_grad_phi[i], first_grad_phi[j] @ Sigma_inv)/(T-1)
                # gg
                J_omega[2*(i)+r,2*(j)+r] = I_omega[2*(i)+r,2*(j)+r]
                # gp
                J_omega[2*(i)+r,2*(j)+r+1] = I_omega[2*(i)+r,2*(j)+r+1]
                # pg
                J_omega[2*(i)+r+1,2*(j)+r] = I_omega[2*(i)+r+1,2*(j)+r]
                # pp
                J_omega[2*(i)+r+1,2*(j)+r+1] = I_omega[2*(i)+r+1,2*(j)+r+1]

    # lmbd - g, eta - g, lmbd - sigma, eta - sigma
    for t in range(X.shape[0]):
        first_grad_g = -np.kron(X[t].ravel('F'), np.eye(N))
        for i in range(r):
            if method == 'ls':
                I_lmbd_g[i] += first_grad_lmbd[i][t] @ Sigma @ first_grad_g/(T-1)
                J_lmbd_g[i] += first_grad_lmbd[i][t] @ first_grad_g/(T-1)
            else:
                I_lmbd_g[i] += first_grad_lmbd[i][t] @ Sigma_inv @ first_grad_g/(T-1)
                J_lmbd_g[i] += first_grad_lmbd[i][t] @ Sigma_inv @ first_grad_g/(T-1)
                # lmbd_sigma
                I_lmbd_sigma[i] -= 0.5*first_grad_lmbd[i][t] @ Sigma_inv @ part_S2 @\
                    np.kron(Sigma_inv,Sigma_inv) @ D /(T-1)  
            
        for j in range(s):
            if method == 'ls':
                # eta - g
                I_eta_g[2*j] += first_grad_gamma[j][t] @ Sigma @ first_grad_g/(T-1)
                I_eta_g[2*j+1] += first_grad_phi[j][t] @ Sigma @ first_grad_g/(T-1)
                J_eta_g[2*j] += first_grad_gamma[j][t] @  first_grad_g/(T-1)
                J_eta_g[2*j+1] += first_grad_phi[j][t] @  first_grad_g/(T-1)
            else:
                # eta - g
                I_eta_g[2*j] += first_grad_gamma[j][t] @ Sigma_inv @ first_grad_g/(T-1)
                I_eta_g[2*j+1] += first_grad_phi[j][t] @ Sigma_inv @ first_grad_g/(T-1)
                J_eta_g[2*j] += first_grad_gamma[j][t] @ Sigma_inv @ first_grad_g/(T-1)
                J_eta_g[2*j+1] += first_grad_phi[j][t] @ Sigma_inv @ first_grad_g/(T-1)

                # eta_sigma
                I_eta_sigma[2*j] -= 0.5*first_grad_gamma[j][t] @ Sigma_inv @ part_S2 @\
                    np.kron(Sigma_inv,Sigma_inv) @ D /(T-1)         
                I_eta_sigma[2*j+1] -= 0.5*first_grad_phi[j][t] @ Sigma_inv @ part_S2 @\
                    np.kron(Sigma_inv,Sigma_inv) @ D /(T-1)        
        # g - sigma
        I_g_sigma -= 0.5*first_grad_g.T @ Sigma_inv @ part_S2 @\
            np.kron(Sigma_inv,Sigma_inv) @ D /(T-1) 


    # I_omega = I_omega + I_omega.T - np.diag(np.diag(I_omega))
    # J_omega = J_omega + J_omega.T - np.diag(np.diag(J_omega))
    I_omega_g = np.vstack((I_lmbd_g, I_eta_g))
    J_omega_g = np.vstack((J_lmbd_g, J_eta_g))
    I_omega_sigma = np.vstack((I_lmbd_sigma, I_eta_sigma))

    return  I_omega, I_omega_g, I_omega_sigma,J_omega,J_omega_g


def asymptotic_g(Sigma,X_AR,X_lmbd,X_eta,N,T,method):
    '''
    return E(d^2L/(dg * dg)): N^2d*N^2d matrix
           E(dL/dg * dL/dg)): N^2d*N^2d matrix (equality)
    '''
    # for g
    # z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
    X = np.concatenate((X_AR, X_lmbd, X_eta), axis = 2)
    Sigma_inv = np.linalg.inv(Sigma)
    I_g = 0
    J_g = 0
    if method == 'ls':
        for t in range(X_AR.shape[0]):
            first_grad = -np.kron(X[t].ravel('F'), np.eye(N))
            I_g += first_grad.T @ Sigma @ first_grad/(T-1)
            J_g += first_grad.T @  first_grad/(T-1)
    else:
        for t in range(X_AR.shape[0]):
            first_grad = -np.kron(X[t].ravel('F'), np.eye(N))
            I_g += first_grad.T @ Sigma_inv @ first_grad/(T-1)
            J_g += first_grad.T @ Sigma_inv @ first_grad/(T-1)
    return I_g,J_g


def elimination_matrix(n):
    # returns L such that vech(A) = L @ vec(A)
    rows, cols = np.tril_indices(n)
    m = len(rows)  # n(n+1)/2
    L = np.zeros((m, n*n), dtype=int)
    for k, (i, j) in enumerate(zip(rows, cols)):
        vec_index = i + j*n  # zero-indexed pos in vec(A) where vec stacks columns
        L[k, vec_index] = 1
    return L

def asymptotic_sigma(Sigma,epsilon,D,T,method):
    '''
    return: I_sigma: E(dL/dsigma * dL/dsigma')   N(N+1)/2*N(N+1)/2 matrix
            J_sigma: E(d^2L/(dsigma * dsigma'))  N(N+1)/2*N(N+1)/2 matrix
    '''
    part_S2 = 0
    N = Sigma.shape[0]
    if method == 'ls':
        L = elimination_matrix(N)
        Sigma_inv = np.eye(N)
        ave_epsilon_outer = np.zeros((N,N))
        for t in range(T-1):
            epsilon_outer = np.outer(epsilon[t],epsilon[t])
            ave_epsilon_outer += epsilon_outer/(T-1)
        part_S2 = 0
        for t in range(T-1):
            epsilon_outer = np.outer(epsilon[t],epsilon[t])
            part_S2 += np.outer(L @ epsilon_outer.ravel('F'), L @ epsilon_outer.ravel('F')) 
        I_sigma =  part_S2/(T-1)-np.outer(L @ ave_epsilon_outer.ravel('F'), L @ ave_epsilon_outer.ravel('F'))
        J_sigma = np.eye(L.shape[0])  # E(d^2L/(dsigma*dsigma))

    else: 
        Sigma_inv = np.linalg.inv(Sigma)
        part_S2 = 0 # E(vec(epsilon * epsilon') vec(epsilon * epsilon')')
        for t in range(T-1):
            epsilon_outer = np.outer(epsilon[t],epsilon[t])
            part_S2 += np.outer(epsilon_outer.ravel('F'), epsilon_outer.ravel('F'))/(T-1)
        Sigma_inv_kron  = np.kron(Sigma_inv, Sigma_inv)
        
        
        I_sigma = 0.25* D.T @ (Sigma_inv_kron @ part_S2 @ Sigma_inv_kron
                                    - np.outer(Sigma_inv.ravel('F'),Sigma_inv.ravel('F')))@ D 
        J_sigma = 0.5* D.T @ Sigma_inv_kron @ D  # E(d^2L/(dsigma*dsigma))
        
    return I_sigma, J_sigma



def asymptotic(lmbd,eta,G,Sigma,y,epsilon, X_AR,X_lmbd,X_eta,p,r,s,P,method):
    '''
    no. of all parameter: b=r+2*s+N^2*d+N(N-1)/2
    Return I: E(dL/dbeta * dL/dbeta') b*b matrix 
           J: E(dL^2/(dbeta*dbeta')) b*b matrix 
    '''
    T,N = y.shape
    d = p+r+2*s
    b = r+2*s+N*N*d + N*(N+1)//2 # no. of all parameter
    # permutation matrix D (N^2 * N(N+1)/2): vec(Sigma) = D vech(Sigma)
    D = np.zeros((N*N,N*(N+1)//2))
    record = 0
    for i in range(N):
        D[i*N+i:(i+1)*N, record : record + (N-i)]=np.eye(N-i)
        record_alt = i
        for j in range(i):
            D[i*N+j,record_alt]=1   
            record_alt  +=  (i-j)
        record =record +  (N-i)

    I = np.zeros((b,b))
    J = np.zeros((b,b))
    
    I_omega, I_omega_g, I_omega_sigma,J_omega,J_omega_g \
        = asymptotic_omega_g_sigma(lmbd,eta,G,Sigma,y,epsilon,X_AR,X_lmbd,X_eta,p,r,s,P,method,D)
    I_g, J_g = asymptotic_g(Sigma,X_AR,X_lmbd,X_eta,N,T,method)
    I_sigma, J_sigma = asymptotic_sigma(Sigma,epsilon,D,T,method)
    
    # omega, omega
    I_omega = I_omega + I_omega.T - np.diag(np.diag(I_omega))
    J_omega = J_omega + J_omega.T - np.diag(np.diag(J_omega))
    I[:r+2*s,:r+2*s]  = I_omega
    J[:r+2*s,:r+2*s]  = J_omega
    
    # g, g
    I[r+2*s:r+2*s+N*N*d,r+2*s:r+2*s+N*N*d] = I_g
    J[r+2*s:r+2*s+N*N*d,r+2*s:r+2*s+N*N*d] = J_g
    
    # sigma, sigma
    I[r+2*s+N*N*d:,r+2*s+N*N*d:] = I_sigma
    J[r+2*s+N*N*d:,r+2*s+N*N*d:] = J_sigma

    # omega, g
    I[:r+2*s,r+2*s:r+2*s+N*N*d] = I_omega_g
    J[:r+2*s,r+2*s:r+2*s+N*N*d] = J_omega_g

    I[r+2*s:r+2*s+N*N*d,:r+2*s] = I_omega_g.T
    J[r+2*s:r+2*s+N*N*d,:r+2*s] = J_omega_g.T
    
    # omega, sigma
    I[:r+2*s,r+2*s+N*N*d:] = I_omega_sigma
    I[r+2*s+N*N*d:,:r+2*s] = I_omega_sigma.T


    return np.linalg.inv(J)@ I @ np.linalg.inv(J)/(T-1)
# np.diag(np.linalg.inv(J)@ I @ np.linalg.inv(J)/(T-1))
# np.diag(J)
# def asymptotic_omega(eta,lmbd,G,Sigma,X2,p,r,s,T):
#     '''
#     return E(d^2L/(domega * domega)): (r+2s)*(r+2s) matrix
#            E(dL/domega * dL/domega)): (r+2s)*(r+2s) matrix (equality)
#     '''
#     power_series= np.arange(1,T-p)

#     I_omega = np.zeros((r+2*s,r+2*s))
#     J_omega = np.zeros((r+2*s,r+2*s))
#     for i in range(r+s):
#         if i < r: # lmbd part
#             lmbd_power_i = np.power(lmbd[i],power_series)
#             lmbd_y_i = np.einsum('i,jki->jki',lmbd_power_i,X2)
#             first_grad_lmbd_i = -G[:,:,p+i] @ np.einsum('i,jki->jk',power_series,(lmbd_y_i/lmbd[i])).T
#             for j in range(i,r+s):
#                 if j < r : # lmbd part
#                     lmbd_power_j = np.power(lmbd[j],power_series)
#                     lmbd_y_j = np.einsum('i,jki->jki',lmbd_power_j,X2)
#                     first_grad_lmbd_j = -G[:,:,p+j] @ np.einsum('i,jki->jk',power_series,(lmbd_y_j/lmbd[j])).T
#                     I_omega[i,j] = np.einsum('ij,ij ->' , first_grad_lmbd_i, Sigma @ first_grad_lmbd_j )/(T-1)
#                     J_omega[i,j] = np.einsum('ij,ij ->' , first_grad_lmbd_i,  first_grad_lmbd_j )/(T-1)
#                 else: # j>=r eta part
#                     gamma_j, phi_j = eta[:,j-r]
#                     gamma_power_j = np.power(gamma_j,power_series)
#                     cos_part_j = np.einsum('i,jki,i->jki',np.cos(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
#                     sin_part_j = np.einsum('i,jki,i->jki',np.sin(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
#                     cos_part_1_j = np.einsum('i,jki->jki',power_series,(cos_part_j/gamma_j)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
#                     sin_part_1_j = np.einsum('i,jki->jki',power_series,(sin_part_j/gamma_j)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
#                     A = G[:,:,p+r+2*(j-r)]  
#                     B = G[:,:,p+r+2*(j-r)+1]
#                     first_grad_gamma_j = -A @ np.sum(cos_part_1_j,axis=2).T - B @ np.sum(sin_part_1_j,axis=2).T
#                     first_grad_phi_j = A @ np.einsum('i,jki->jk',power_series,(sin_part_j)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_j)).T
#                     I_omega[i,r+2*(j-r)] = np.einsum('ij,ij->',first_grad_lmbd_i ,Sigma @ first_grad_gamma_j )/(T-1)
#                     I_omega[i,r+2*(j-r)+1] = np.einsum('ij,ij->',first_grad_lmbd_i ,Sigma @ first_grad_phi_j )/(T-1)
#                     J_omega[i,r+2*(j-r)] = np.einsum('ij,ij->',first_grad_lmbd_i , first_grad_gamma_j )/(T-1)
#                     J_omega[i,r+2*(j-r)+1] = np.einsum('ij,ij->',first_grad_lmbd_i , first_grad_phi_j )/(T-1)
#         else: # i>=r eta part
#             gamma_i, phi_i = eta[:,i-r]
#             gamma_power_i = np.power(gamma_i,power_series)
#             cos_part_i = np.einsum('i,jki,i->jki',np.cos(phi_i*power_series), X2, gamma_power_i) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
#             sin_part_i = np.einsum('i,jki,i->jki',np.sin(phi_i*power_series), X2, gamma_power_i) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
#             cos_part_1_i = np.einsum('i,jki->jki',power_series,(cos_part_i/gamma_i)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
#             sin_part_1_i = np.einsum('i,jki->jki',power_series,(sin_part_i/gamma_i)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
#             A = G[:,:,p+r+2*(i-r)]  
#             B = G[:,:,p+r+2*(i-r)+1]
#             first_grad_gamma_i = -A @ np.sum(cos_part_1_i,axis=2).T - B @ np.sum(sin_part_1_i,axis=2).T
#             first_grad_phi_i = A @ np.einsum('i,jki->jk',power_series,(sin_part_i)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_i)).T
#             for j in range(i,r+s):
#                 gamma_j, phi_j = eta[:,j-r]
#                 gamma_power_j = np.power(gamma_j,power_series)
#                 cos_part_j = np.einsum('i,jki,i->jki',np.cos(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
#                 sin_part_j = np.einsum('i,jki,i->jki',np.sin(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
#                 cos_part_1_j = np.einsum('i,jki->jki',power_series,(cos_part_j/gamma_j)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
#                 sin_part_1_j = np.einsum('i,jki->jki',power_series,(sin_part_j/gamma_j)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
#                 A = G[:,:,p+r+2*(j-r)]  
#                 B = G[:,:,p+r+2*(j-r)+1]
#                 first_grad_gamma_j = -A @ np.sum(cos_part_1_j,axis=2).T - B @ np.sum(sin_part_1_j,axis=2).T
#                 first_grad_phi_j = A @ np.einsum('i,jki->jk',power_series,(sin_part_j)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_j)).T
#                 if i == j:
#                     # gg
#                     I_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_gamma_j)/(T-1)
#                     # gp
#                     I_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_phi_j)/(T-1)
#                     # pp
#                     I_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i,Sigma @ first_grad_phi_j)/(T-1)
#                     # gg
#                     J_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_gamma_j)/(T-1)
#                     # gp
#                     J_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_phi_j)/(T-1)
#                     # pp
#                     J_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i, first_grad_phi_j)/(T-1)
                    
#                 else: # i =! j
#                     # gg
#                     I_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_gamma_j)/(T-1)
#                     # gp
#                     I_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i,Sigma @ first_grad_phi_j)/(T-1)
#                     # pg
#                     I_omega[2*(i-r)+r+1,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_phi_i,Sigma @ first_grad_gamma_j)/(T-1)
#                     # pp
#                     I_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i,Sigma @ first_grad_phi_j)/(T-1)
#                     # gg
#                     J_omega[2*(i-r)+r,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_gamma_j)/(T-1)
#                     # gp
#                     J_omega[2*(i-r)+r,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_gamma_i, first_grad_phi_j)/(T-1)
#                     # pg
#                     J_omega[2*(i-r)+r+1,2*(j-r)+r] = np.einsum('ij,ij->',first_grad_phi_i, first_grad_gamma_j)/(T-1)
#                     # pp
#                     J_omega[2*(i-r)+r+1,2*(j-r)+r+1] = np.einsum('ij,ij->',first_grad_phi_i, first_grad_phi_j)/(T-1)
                   
#     I_omega = I_omega + I_omega.T - np.diag(np.diag(I_omega))
#     J_omega = J_omega + J_omega.T - np.diag(np.diag(J_omega))
#     return  I_omega,J_omega


# def asymptotic_g(Sigma,z,N,T):
#     '''
#     return E(d^2L/(dg * dg)): N^2d*N^2d matrix
#            E(dL/dg * dL/dg)): N^2d*N^2d matrix (equality)
#     '''
#     # for g
#     # z = kron([L[:T-1,:].T,np.identity(N)]) @ X1
#     I_g = 0
#     J_g = 0
#     for t in range(T-1):
#         first_grad = -np.kron(z[:,t], np.eye(N))
#         I_g += first_grad.T @ Sigma @ first_grad/(T-1)
#         J_g += first_grad.T @  first_grad/(T-1)
#     return I_g,J_g



# def asymptotic_omega_g(lmbd,eta,G,Sigma,z,X2,p,r,s,N,T):
#     '''
#     return E(d^2L/(domega * dg)): (r+2s)*(N^2 d) matrix
#     '''
#     power_series= np.arange(1,T-p)
#     d = p+r+2*s
#     # lmbd_g
#     I_lmbd_g = np.zeros((r, N*N*d))
#     J_lmbd_g = np.zeros((r, N*N*d))
#     for i in range(r):
#         lmbd_power_i = np.power(lmbd[i],power_series)
#         lmbd_y_i = np.einsum('i,jki->jki',lmbd_power_i,X2)
#         first_grad_lmbd_i = -G[:,:,p+i] @ np.einsum('i,jki->jk',power_series,(lmbd_y_i/lmbd[i])).T
#         for t in range(T-1):
#             first_grad_g = -np.kron(z[:,t], np.eye(N))
#             I_lmbd_g[i] += first_grad_lmbd_i[:,t] @ Sigma @ first_grad_g/(T-1)
#             J_lmbd_g[i] += first_grad_lmbd_i[:,t] @ first_grad_g/(T-1)
#     # eta_g
#     I_eta_g = np.zeros((2*s,N*N*d))
#     J_eta_g = np.zeros((2*s,N*N*d))
#     for j in range(s):
#         gamma_j, phi_j = eta[:,j]
#         gamma_power_j = np.power(gamma_j,power_series)
#         cos_part_j = np.einsum('i,jki,i->jki',np.cos(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) cos{(j-p)phi} y_{t-j}
#         sin_part_j = np.einsum('i,jki,i->jki',np.sin(phi_j*power_series), X2, gamma_power_j) #gamma^(j-p) sin{(j-p)phi} y_{t-j}
#         cos_part_1_j = np.einsum('i,jki->jki',power_series,(cos_part_j/gamma_j)) # (j-p) gamma^(j-p-1) cos{(j-p)phi} y_{t-j}
#         sin_part_1_j = np.einsum('i,jki->jki',power_series,(sin_part_j/gamma_j)) # (j-p) gamma^(j-p-1) sin{(j-p)phi} y_{t-j}
#         A = G[:,:,p+r+2*j] 
#         B = G[:,:,p+r+2*j+1]
#         first_grad_gamma_j = -A @ np.sum(cos_part_1_j,axis=2).T - B @ np.sum(sin_part_1_j,axis=2).T
#         first_grad_phi_j = A @ np.einsum('i,jki->jk',power_series,(sin_part_j)).T - B @ np.einsum('i,jki->jk',power_series,(cos_part_j)).T
#         for t in range(T-1):
#             first_grad_g = -np.kron(z[:,t], np.eye(N))
#             I_eta_g[2*j] += first_grad_gamma_j[:,t] @ Sigma @ first_grad_g/(T-1)
#             I_eta_g[2*j+1] += first_grad_phi_j[:,t] @ Sigma @ first_grad_g/(T-1)
#             J_eta_g[2*j] += first_grad_gamma_j[:,t] @  first_grad_g/(T-1)
#             J_eta_g[2*j+1] += first_grad_phi_j[:,t] @  first_grad_g/(T-1)
#     return np.vstack((I_lmbd_g,I_eta_g)),np.vstack((J_lmbd_g,J_eta_g))




# def asymptotic(lmbd,eta,G,Sigma,z,X2,epsilon,p,r,s,N,T):
#     '''
#     no. of all parameter: b=r+2*s+N^2*d+N(N-1)/2
#     Return I: E(dL/dbeta * dL/dbeta') b*b matrix 
#            J: E(dL^2/(dbeta*dbeta')) b*b matrix 
#     '''
#     d = p+r+2*s
#     b = r+2*s+N*N*d # no. of all parameter
#     # permutation matrix D (N^2 * N(N+1)/2): vec(Sigma) = D vech(Sigma)
#     # D = np.zeros((N*N,N*(N+1)//2))
#     # record = 0
#     # for i in range(N):
#     #     D[i*N+i:(i+1)*N, record : record + (N-i)]=np.eye(N-i)
#     #     record_alt = i
#     #     for j in range(i):
#     #         D[i*N+j,record_alt]=1   
#     #         record_alt  +=  (i-j)
#     #     record =record +  (N-i)
        
#     I = np.zeros((b,b))
#     J = np.zeros((b,b))
#     # omega, omega
#     I[:r+2*s,:r+2*s]  = asymptotic_omega(eta,lmbd,G,Sigma,X2,p,r,s,T)[0]
#     J[:r+2*s,:r+2*s]  = asymptotic_omega(eta,lmbd,G,Sigma,X2,p,r,s,T)[1]
    
#     # g, g
#     I[r+2*s:r+2*s+N*N*d,r+2*s:r+2*s+N*N*d] = asymptotic_g(Sigma,z,N,T)[0]
#     J[r+2*s:r+2*s+N*N*d,r+2*s:r+2*s+N*N*d] = asymptotic_g(Sigma,z,N,T)[1]
    
#     # # sigma, sigma
#     # I[r+2*s+N*N*d:,r+2*s+N*N*d:] = asymptotic_sigma(Sigma_inv,epsilon,D,T)[0]
#     # J[r+2*s+N*N*d:,r+2*s+N*N*d:] = asymptotic_sigma(Sigma_inv,epsilon,D,T)[1]

#     # omega, g
#     I[:r+2*s,r+2*s:r+2*s+N*N*d] = asymptotic_omega_g(lmbd,eta,G,Sigma,z,X2,p,r,s,N,T)[0]
#     J[:r+2*s,r+2*s:r+2*s+N*N*d] = asymptotic_omega_g(lmbd,eta,G,Sigma,z,X2,p,r,s,N,T)[1]
#     I[r+2*s:r+2*s+N*N*d,:r+2*s] = I[:r+2*s,r+2*s:r+2*s+N*N*d].T
#     J[r+2*s:r+2*s+N*N*d,:r+2*s] = J[:r+2*s,r+2*s:r+2*s+N*N*d].T
    
#     # # alpha, sigma
#     # I[:r+2*s+N*N*d,r+2*s+N*N*d:] = asymptotic_alpha_sigma(lmbd,eta,G,Sigma_inv,z,X2,epsilon,D,p,r,s,N,T)
#     # I[r+2*s+N*N*d:,:r+2*s+N*N*d] = I[:r+2*s+N*N*d,r+2*s+N*N*d:].T

#     # return Gamma,Sigma
#     return np.linalg.inv(J)@ I @ np.linalg.inv(J)/(T-1)

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