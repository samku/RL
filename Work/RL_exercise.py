import numpy as np
import matplotlib.pyplot as plt
import SMD as sys

#Positive angle is to the right
system = sys.SMD()
Ts= 0.0001

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

#Control matrix
u = np.matrix(np.zeros((nu,N+1)))

#Initial H matrix
#Q(x,u) = [x u][H11 H12][x]
#              [H21 H22][u]
H11 = np.matrix(0.001*np.ones((nx,nx)))
H12 = np.matrix(0.001*np.ones((nx,nu)))
H21 = np.matrix(0.001*np.ones((nu,nx)))
H22 = np.matrix(0.001*np.ones((nu,nu)))
H = np.concatenate((np.concatenate((H11,H12),axis=1),np.concatenate((H21,H22),axis=1)),axis=0)

#Cost matrices for LQR
E = np.matrix(np.eye(nx,nx))*1000
#E[2,2] = 0.
#E[3,3] = 0.
#print E
F = np.matrix(np.eye(nu,nu))*0

#Learning parameters
gamma = 0.01 #Discount factor
alpha = 0.00001 #Hessian approx

#Simulate one episode of system
for episodes in range(1,20):
    #Calculate feedback gain
    U = np.matrix(-1.*np.linalg.inv(H22)*H21)
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
        #Calculate gradients of Q wrt H matrix
        dQH11 = x[:,i-1]*np.transpose(x[:,i-1])
        dQH12 = x[:,i-1]*np.transpose(u[:,i-1])
        dQH21 = u[:,i-1]*np.transpose(x[:,i-1])
        dQH22 = u[:,i-1]*np.transpose(u[:,i-1])
        dQH = np.concatenate((np.concatenate((dQH11,dQH12),axis=1),np.concatenate((dQH21,dQH22),axis=1)),axis=0)
        #Calculate correction in H
        dH = alpha*(r_xu+gamma*Q_nxt-Q_curr)*dQH
        H = H+dH
        #Extract individual updated H pieces
        H11 = H[0:nx,0:nx]
        H12 = H[0:nx,nx:]
        H21 = H[nx:,0:nx]
        H22 = H[nx:,nx:]
    print H
    plt.plot(np.transpose(x[0,:]))
    plt.pause(0.001)

plt.show()