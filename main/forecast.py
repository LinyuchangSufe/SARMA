import numpy as np
from tensorly.tenalg import mode_dot # tensor multiply
from main.help_function_for_LS import * 
from main.tensorOp import * 

def forecast_SARMA(y,p,r,s,lmbd,gamma,phi, G, h=1):
    '''
    forecast h step prediction
    '''
    N,T = y.shape
    y_pre = np.zeros((N,h)) ## save forecasts
    eta = np.vstack((gamma,phi))
    L = get_L(lmbd,eta,r,s,T+h-1,p) # L: T*d matrix
    A = mode_dot(G,L,2) # A: N*N*(T+h-1) matrix
    
    

    for k in range(h):
        # useful preparation for pre
        x = np.concatenate((np.flip(y_pre[:,:k],axis=1).ravel(order='F'), 
             (np.flip(y,axis=1)).ravel(order='F'))) # NT array: vectorized from y_T to y_1
        y_pre[:,k] = tensor_op.unfold(A[:,:,:T+k],0).numpy() @x
    
    return y_pre