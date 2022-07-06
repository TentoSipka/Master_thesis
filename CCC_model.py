import numpy as np
from arch import arch_model

def run_univariate_garch(ret):
    model_params = {}
    T,N = ret.shape
    
    D= np.zeros((T,N))
    for r_i in ret:
        am = arch_model (ret[r_i])
        model_params[r_i] = am.fit(disp='off')
        D[:,ret.columns.get_loc(r_i)] =model_params[r_i].conditional_volatility
    return  model_params, D

def ccceq(ret, D_inv,init):  

    T,N = ret.shape  
    
    Eps = np.zeros((T,N))    
    for t in range(T):
        Eps[t,:] = np.matmul(np.diagflat(D_inv[t,:]), ret[t])  
     
    Rt = np.zeros((N, N ,T))  
    
    for j in range(0,min(init,N)):      
        Rt[:,:,j] = np.cov(Eps[:init,:].T) 
  

    for j in range(min(init,N),T):      
        Rt[:,:,j] = np.cov(Eps[:(j-1),:].T) 
    return Rt         
