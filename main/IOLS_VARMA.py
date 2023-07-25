import numpy as np
def IOLS_pre(y):
    N,T = y.shape
    P = int(np.log(T))
    # create X (regressors)
    X = np.zeros((N*P,T-P))
    for i in range(P):
        X[i*N:i*N+N,:] = y[:,P-i-1:T-i-1]
    # create Y (response)
    Y = y[:,P:]
    A = (X@Y.T).T @ np.linalg.inv(X @ X.T)
    # solve OLS
    U_fit = np.zeros((N,T))
    U_fit[:,P:] = Y - A@X # N*(T-P)
    for k in range(100):
        Z = np.zeros((N*2,T-1))
        Z[:N,:] = y[:,1-1:T-1]
        Z[N:2*N,:] = U_fit[:,1-1:T-1]
        Y = y[:,1:]
        A_new = (Z@Y.T).T @ np.linalg.inv(Z @ Z.T)
        U_after = np.zeros((N,T))
        U_after[:,1:] = Y - A_new@Z
        U_diff = np.linalg.norm(U_fit - U_after, ord='fro')
        U_fit = U_after
        if U_diff < 1e-7:
            break
    y_pre = A_new @ np.concatenate((y[:,-1:] , U_fit [:,-1:]))

    return y_pre