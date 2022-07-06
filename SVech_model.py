import numpy as np 

def svech_est_H(theta,ret,init,intercept_m= None):
    

    T,N = np.shape(ret)
    a,b = theta
    Ht = np.zeros((N, N ,T))
    
    if intercept_m is None:
        intercept_m = np.zeros((N,N))
        for t in range(T):
            intercept_m= intercept_m +np.matmul(np.array(ret[t,:], ndmin=2).T,np.array(ret[t,:], ndmin=2))
        intercept_m= 1/T * intercept_m  
   
    Ht[:,:,0] = np.cov(ret[:init,:].T) 

    for j in range(1,T):      
        Ht[:,:,j] = (1-a-b) * intercept_m  
        Ht[:,:,j] = Ht[:,:,j] + a *np.matmul(np.array(ret[j-1,:], ndmin=2).T,np.array(ret[j-1,:], ndmin=2))
        Ht[:,:,j] = Ht[:,:,j] + b * Ht[:,:,j-1]
        
      
    return Ht  


def eq_svech_svd(theta,ret,init,intercept_m, SVD = False):
    
    T,N = np.shape(ret)
    a,b = theta
    Ht = np.zeros((N, N ,T)) 
    U = np.zeros((N, N ,T)) 
    s = np.zeros((N ,T)) 
    Vh = np.zeros((N, N ,T)) 
    
    ret_shifted = np.insert(ret, 0, ret[0,:], axis=0)
    ret_shifted = np.delete(ret_shifted,obj = -1, axis =0)

    Ht[:,:,0] = np.cov(ret[:init,:].T) 
    if SVD == True:
        U[:,:,0],s[:,0],Vh[:,:,0] = np.linalg.svd(Ht[:,:,0], full_matrices=True, compute_uv=True, hermitian=True)
    
    for n in range(0,N):
        for m in range(0,N): 
            Ht[n,m,:] = (1-a-b) * intercept_m[n,m]+ a * ret_shifted[:,n]*ret_shifted[:,m] 
   
    Ht[:,:,0] = np.cov(ret[:init,:].T) 
    for t in range(1,T):
        Ht[:,:,t] = Ht[:,:,t] +  b* Ht[:,:,t-1]
        if  SVD == True:
            U[:,:,t],s[:,t],Vh[:,:,t] = np.linalg.svd(Ht[:,:,t], full_matrices=True, compute_uv=True, hermitian=True)
           
    if SVD==True:
        return U,s,Vh   
    else: 
        return Ht
    
def loglike_svech(theta,ret,init, intercept_m = None, SVD= False, verbose = False):
    
    T,N = np.shape(ret)
    llf = np.zeros((T,1))
    outs =[]
    if intercept_m is None:
        intercept_m = np.zeros((N,N))
        for t in range(T):
            intercept_m= intercept_m +np.matmul(np.array(ret[t,:], ndmin=2).T,np.array(ret[t,:], ndmin=2))
        intercept_m= 1/T * intercept_m  
    
    if SVD == True:
        U,s,Vh  =  eq_svech_svd(theta,ret,init,intercept_m, SVD) 
    else:
        Ht =   eq_svech_svd(theta,ret,init,intercept_m, SVD) 
 
    
    if SVD == True:  
        for i in range(0,T):      

            rV = np.matmul(np.array(ret[i,:], ndmin=2) , Vh[:,:,i].T)
            Ur = np.matmul(U[:,:,i].T, np.array(ret[i,:], ndmin=2).T)
            rVS = np.divide(rV, s[:,i]) 
            llf[i] = np.matmul(rVS, Ur)

        llf = np.sum(llf) + s.sum()
    else:  
        for i in range(0,T):
            det = np.log(np.linalg.det(Ht[:,:,i]))            
            mult = np.matmul(np.matmul(np.array(ret[i,:], ndmin=2) , (np.linalg.inv(Ht[:,:,i]))) ,np.array(ret[i,:], ndmin=2).T)
            llf[i] =(det+ mult)
        llf = np.sum(llf)
    if verbose:
        print(theta,llf)
        
    if llf == float("-inf"): #trick to keep alhorithom going even when singular matrix is obtained in some interation
        llf=0
       
    return llf


def composite_pair_loglike_svech(theta,ret,init):
    T,N = np.shape(ret)
    Hjt= np.zeros((N, N ,T)) 
    llfj = np.zeros((T,1))
    
    ňintercept_m = np.zeros((N,N))
    intercept_m= 1/T*np.matmul(ret.T,ret)
        
    Hj =   eq_svech_svd(theta,ret,init,intercept_m, False)  

    Hj_inv = np.zeros((N,N,T))  
    det = np.zeros((T))
    det[:] =   Hj[0,0,:]* Hj[1,1,:]- Hj[0,1,:]* Hj[1,0,:]    
    Hj_inv[0,0,:] =  Hj[1,1,:] /det[:]
    Hj_inv[1,1,:] =  Hj[0,0,:] /det[:]
    Hj_inv[0,1,:] =  -Hj[0,1,:] /det[:]
    Hj_inv[1,0,:] =  -Hj[1,0,:] /det[:]   
   
    mult  = np.einsum('ij, jki, ik ->', ret, Hj_inv, ret)   
    llfj = np.sum(np.log(det))+np.sum(mult) 
    llfj = np.sum(llfj)  
        
    return llfj     
    
def composite_loglike(theta,ret,init, endpair= None, verbose = False):
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
                llf[i]= composite_pair_loglike_svech(theta,retj,init)
                i+=1
    llf= np.sum(llf)
    if verbose:
        print(theta,llf)   
            
    return llf
    