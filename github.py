# https://github.com/AaronvonMaekel/UncertaintyQuantification/blob/9e2d2bfeb090763183e226c9de7216e04a5d2a9e/eulerML.py#L31

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
u_zero = [0.5,2]
T = 6
delta = 0.2
hmesh = [0.6,0.06,0.006,0.0006] # lieber duplzieren statts mal 10
Nmesh = np.logspace(2,4,2,dtype=int)
results = np.empty((len(hmesh),len(Nmesh)))
results_var = np.empty((len(hmesh),len(Nmesh)))
def f(u):
    return np.array([ u[0] - u[0]*u[1], u[0]*u[1] - u[1]])

def calc_Q_est(N,h,var=False):
    global u_zero,T
    Q=[]
      
    for i in range(int(N)):
        u_noise = u_zero + random.uniform(low=-delta, high=delta, size=2)

        for t in range(int(T/h)):
            u_noise = u_noise + h*f(u_noise)
        Q.append(u_noise[0]) 
    Q_est = np.mean(Q)
    if var:
        Q_var = np.sum((Q - Q_est)**2)/(N/1)
        return Q_est,Q_var
    else:
        return Q_est

def calc_Y_est(N,h1,h2,var=False):
    global u_zero,T
    Y=[]
      
    for i in range(int(N)):
        u_noise1 = u_zero + random.uniform(low=-delta, high=delta, size=2)
        u_noise2 = u_noise1
        for t in range(int(T/h1)):
            u_noise1 = u_noise1 + h1*f(u_noise1)
        for t in range(int(T/h2)):
            u_noise2 = u_noise2  + h2*f(u_noise2)
        Y.append(u_noise1[0] - u_noise2[0]) 
    Y_est = np.mean(Y)
    if var:
        Y_var = np.sum((Y - Y_est)**2)/(N/1)
        return Y_est,Y_var
    else:
        return Y_est


def MLMC(max_L,Nls,hls,log = False):
    Q_est = calc_Q_est(Nls[0],hls[0])
    if log:
        Y_est=[]
        Y_est.append(Q_est)
    for l in range(1,max_L):
        Y = (calc_Y_est(Nls[l],hls[l],hls[l-1]))
        Q_est +=  Y
        if log:
            Y_est.append(Y)
    if log:
        return Q_est,Y_est
    else:    
        return Q_est


def adaptive_MLMC(epsilon):
    h0 =0.1
    m = 2
    L = 1
    N_init = 10000
    N = []
    r=0.9
    gamma=1
    N.append( N_init)
    N.append( N_init)
    #do first loop
    
    h1=h0/m
    cycle=1
    while(True):
        print(cycle)
        Y = []
        sl = []
        C=[]
        y,s = calc_Q_est(N[0],h0,True)
        C.append(h0**(-gamma))
        Y.append(y)
        sl.append(s)
        alpha=1
        h1=h0
        for l in range(L+1):
            y,s = calc_Y_est(N[l],h1/m,h1,True)
            C.append(h1**(-gamma))
            h1/=m
            Y.append(y)
            sl.append(s)
        C = np.array(C)
        sl = np.array(sl)
        #update N
        konstant = 2/(epsilon**2)*(np.sum(np.sqrt(C*sl)))
        N = np.ceil(konstant *np.sqrt(sl/C)).tolist()
        
        if Y[-1] > (epsilon *r*m**alpha -1)/np.sqrt(2):
            N.append(N_init)
            L+=1
        else:
            if np.sum(sl/np.array(N))< (epsilon**2)/2:
                break
        cycle+=1
    Q_est = np.sum(Y_est)
    return(Q_est)
            

slide_Q = 1.3942

if __name__ == "__main__":
    
    print(adaptive_MLMC(0.1))
    ######## a)
    L= 1
    N5 = 1000*np.logspace(0,-L+1,L,base=2)
    h5 = 0.03 *np.logspace(0,-L+1,L,base=2)
    
    Q_est,Y_est = MLMC(L,N5,h5,log=True) 
        # Plotting the curves
    plt.figure(figsize=(10, 6))
    print(Y_est)
    print("final Q estimator:", np.sum(Y_est))
    plt.plot(range(0,L),np.abs( Y_est),  color='red')
    plt.yscale("log")
    
    plt.xlabel('Level')
    plt.ylabel('Y_est')
   
    plt.grid()
    # Show plot
    plt.show()
    