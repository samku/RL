import sys
sys.path.append(r"casadiinstalldir")
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import SMD as sys

#Positive angle is to the right
system = sys.SMD()
Ts= 0.001

#Sizes of states and inputs - Assuming full state feedback
nx = system.nx
nu = system.nu

#Episode length
N = 10000

#State matrix
x = np.matrix(np.zeros((nx,N+1)))
#Initial state
#x0 = np.matrix(([np.pi/100],[0.],[0.],[0.]))
x0 = np.matrix(([10.],[0.]))
x[:,0] = x0
print np.shape(x)

#Control matrix
u = np.matrix(np.zeros((nu,N+1)))

#H matrix - Make this a casadi variable
#Q(x,u) = [x u][H11 H12][x]
#              [H21 H22][u]
#H11 = MX.sym("H11",nx,nx)
#H12 = MX.sym("H12",nx,nu)
#H21 = MX.sym("H21",nu,nx)
#H22 = MX.sym("H22",nu,nu)
H = MX.sym("H",nx+nu, nx+nu)
#H = np.concatenate((np.concatenate((H11,H12),axis=1),np.concatenate((H21,H22),axis=1)),axis=0)

#Cost matrices for LQR
E = np.matrix(np.eye(nx,nx))
F = np.matrix(np.eye(nu,nu))*100

#Learning parameters
gamma = 1.0 #Discount factor
alpha = 0.0005 #Hessian approx

#Initial values
H22 = 0.1*np.matrix(np.ones((nu,nu)))
H21 = 0.1*np.matrix(np.ones((nu,nx)))

#Simulate one episode of system
for episodes in range(1,2):
    #Calculate feedback gain
    U = np.matrix(-1.*np.linalg.inv(H22)*H21)
    print U
    for i in range(1,N):
        #Calculate control input in read state + exploration noise
        u[:,i-1] = U*x[:,i-1]+0.001*np.random.rand(1)
        #Propogate system to next state
        x[:,i] = system.state_update(x[:,i-1],u[:,i-1],Ts)
        #Calculate LQR stage cost = x'Ex + u'Fu
        r_xu = np.asscalar(np.transpose(x[:,i-1])*E*x[:,i-1]+np.transpose(u[:,i-1])*F*u[:,i-1])
        #Calculate future cost (TD(0)) from next state following U (Using previous iteration H)
        u[:,i] = U*x[:,i]
        X_nxt = np.concatenate((x[:,i],u[:,i]),axis=0)
        Q_nxt = np.asscalar(np.transpose(X_nxt)*H*X_nxt)
        #Calculate current expected total cost (Using previous iteration H)
        X_curr = np.concatenate((x[:,i-1],u[:,i-1]),axis=0)
        Q_curr = np.asscalar(np.transpose(X_curr)*H*X_curr)
        #We have r_xu = Q_curr + gamma*Q_nxt
        #r_xu is a scalar number, Q_curr and Q_nxt are the same in terms of H. 
        #At the end of each episode, H is calculated in an LS sense
        print Q_nxt
    #Each episode end, the calculated H values give one policy iteration step

plt.show()