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

# calculate R_t and Q as in quation (1) return Rt and vech(R_t) - lower part of matrix as stacked vector 
def dcceq(phi,ret,D_inv,init, Eps = None):  

    a, b = phi
    T,N = ret.shape   
    
    if Eps is None:
        Eps = np.zeros((T,N))    
        for t in range(T):
            Eps[t,:] = np.matmul(np.diagflat(D_inv[t,:]), ret[t])  
      
    S= 1/T*np.matmul(Eps.T,Eps) 

    if min(a,b)<0 or max(a,b)>1 or a+b > .999999:
        a = .9999 - b
        
    Qt = np.zeros((N, N ,T))
    Qt[:,:,0] = np.cov(Eps[:init,:].T) 
    
    Eps_shifted = np.insert(Eps, 0, Eps[0,:], axis=0)
    Eps_shifted = np.delete(Eps_shifted,obj = -1, axis =0)
   
    Rt =  np.zeros((N, N ,T))     
    Rt[:,:,0] = np.corrcoef(Eps[:init,:].T)

     
    for n in range(0,N):
        for m in range(0,N):     
            Qt[n,m,:] = S[n,m] * (1-a-b)
            Qt[n,m,:] = Qt[n,m,:] + a * Eps_shifted[:,n]*Eps_shifted[:,m]
    Qt[:,:,0] = np.cov(Eps[:init,:].T)
    
    for t in range(1,T):
        Qt[:,:,t] = Qt[:,:,t] +  b* Qt[:,:,t-1]
    

    for n in range(0,N):
        for m in range(0,N):
             Rt[n,m,:] = np.divide(Qt[n,m,:],np.sqrt(Qt[n,n,:]*Qt[m,m,:]))
        
    return Rt           



def loglike_dcc_svd(phi,ret,D_inv,init, SVD= False, verbose= False):
    
    T,N = ret.shape  
    llf = np.zeros((T,1))    
    
    Eps = np.zeros((T,N))    
    for t in range(T):
        Eps[t,:] = np.matmul(np.diagflat(D_inv[t,:]), ret[t])  
    
    Rt =  dcceq(phi,ret,D_inv,init,Eps)
    
    U = np.zeros((N, N ,T)) 
    s = np.zeros((N ,T)) 
    Vh = np.zeros((N, N ,T)) 
    
        
    if SVD == True:
        for j in range(0,T):
            U[:,:,j],s[:,j],Vh[:,:,j]= np.linalg.svd(Rt[:,:,j], full_matrices=True, compute_uv=True, hermitian=True)

     
    if SVD == True:
        for i in range(0,T):               
            rV = np.matmul(np.array(Eps[i,:], ndmin=2) , Vh[:,:,i].T)
            Ur = np.matmul(U[:,:,i].T, np.array(Eps[i,:], ndmin=2).T)
            rVS = np.divide(rV, s[:,i]) 
            det = np.log(s[:,i]).sum()
            mult = np.matmul(rVS, Ur)- np.matmul(np.array(Eps[i,:], ndmin=2), np.array(Eps[i,:], ndmin=2).T) #Make sure this makes sense
            llf[i] =det + mult
        llf = np.sum(llf) 
    else:
        for i in range(0,T):
            det =  np.log(np.linalg.det(Rt[:,:,i]))
            mult =   np.matmul(np.matmul(Eps[i,:].T , (np.linalg.inv(Rt[:,:,i]) - np.eye(N))) ,Eps[i,:])
            llf[i] =(det+ mult)  
        llf = np.sum(llf)
   
    if verbose:
        print(phi,llf)  
       
    return llf


def composite_pair_loglike_dcc(phi,ret,D_inv,init):
    T,N = np.shape(ret)
    Rj = np.zeros((N, N ,T)) 
    llfj = np.zeros((T,1))
    
    Eps = np.zeros((T,N))    
    for t in range(T):
        Eps[t,:] = D_inv[t,:]* ret[t]    
    Eps_sq = Eps**2
    Rj =  dcceq(phi,ret,D_inv,init, Eps)   
    
    Rj_inv = np.zeros((N,N,T))  
    det = np.zeros((T))
    det[:] =   Rj[0,0,:]* Rj[1,1,:]- Rj[0,1,:]* Rj[1,0,:]    
    Rj_inv[0,0,:] =  Rj[1,1,:] /det[:]
    Rj_inv[1,1,:] =  Rj[0,0,:] /det[:]
    Rj_inv[0,1,:] =  -Rj[0,1,:] /det[:]
    Rj_inv[1,0,:] =  -Rj[1,0,:] /det[:]   
   
    Eps_sq_col_sum = Eps_sq.sum(axis = 1)   

    mult  = np.einsum('ij, jki, ik ->', Eps, Rj_inv, Eps)   
    mult_total =  mult - np.sum(Eps_sq_col_sum)   
    llfj = np.sum(np.log(det))+np.sum(mult_total)  
       
    return llfj     
    
def composite_loglike(phi,ret,D_inv,init, endpair= None, verbose = False):
    T,N = np.shape(ret)
    llf = np.zeros((int(N*(N-1)/2),1))
    i=0
    
    if endpair is not None:
        endpair = min(N,endpair)
        endpair = max(2,endpair)
    else:
        endpair = N
        
    for m in range(1,endpair):
        for n in range(0,N):
            if m+n < N:
                retj =ret[:,[n,m+n]]
                D_invj = D_inv[:,[n,m+n]]
                llf[i]= composite_pair_loglike_dcc(phi,retj,D_invj,init)
                i+=1
    llf= np.sum(llf)
    
    if verbose:
        print(phi,llf) 
      
    if llf == float("-inf"): #trick to keep alhorithom going even when singular matrix is obtained in some interation
        llf=0              
    
    return llf