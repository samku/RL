import numpy as np

class SMD:
    def __init__(self):
        #Pendulum cart parameters
        self.M = 2.
        self.c = 10.
        self.K = 10.
        self.nx = 2
        self.nu = 1

    def state_space(self,x0,u):
        #Extract parameters
        M = self.M
        c = self.c
        K = self.K
        #Extract variables
        x1 = x0[0]
        x2 = x0[1]
        # [x,dx]
        dx = np.matrix(np.zeros((2,1)))
        dx[0] = x2
        dx[1] = (u-c*x2-K*x1)/M
        return dx

    def state_update(self,x0,u,Ts):
        #RK Integration
        k1 = Ts*self.state_space(x0,u)
        k2 = Ts*self.state_space(x0+k1/2,u)
        k3 = Ts*self.state_space(x0+k2/2,u)
        k4 = Ts*self.state_space(x0+k3,u)
        return x0+(k1/6)+(k2/3)+(k3/3)+(k4/6)








