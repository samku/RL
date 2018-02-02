import numpy as np

class PendulumCart:
    def __init__(self):
        #Pendulum cart parameters
        self.M = 2.4
        self.m = 0.23
        self.l = 0.36
        self.g = 9.81
        self.nx = 4
        self.nu = 1

    def state_space(self,x0,u):
        #Extract parameters
        M = self.M
        m = self.m
        l = self.l
        g = self.g
        #Extract variables
        x1 = x0[0]
        x2 = x0[1]
        x3 = x0[2]
        x4 = x0[3]
        # [theta,dtheta,x,dx]
        dx = np.matrix(np.zeros((4,1)))
        dx[0] = x2
        dx[1] = (u*np.cos(x1)-(M+m)*g*np.sin(x1) + \
                     m*l*np.cos(x1)*np.sin(x1)*np.power(x2,2))/(m*l*np.power(np.cos(x1),2)-(M+m)*l)
        dx[2] = x4
        dx[3] = (u + m*l*np.sin(x1)*np.power(x2,2)-m*g*np.cos(x1)*np.sin(x1))/(M+m-m*np.power(np.cos(x1),2))
        return dx

    def state_update(self,x0,u,Ts):
        #RK Integration
        k1 = Ts*self.state_space(x0,u)
        k2 = Ts*self.state_space(x0+k1/2,u)
        k3 = Ts*self.state_space(x0+k2/2,u)
        k4 = Ts*self.state_space(x0+k3,u)
        return x0+(k1/6)+(k2/3)+(k3/3)+(k4/6)

#system = PendulumCart()
#x0 = np.array([np.pi/100, 0., 0., 0.])
#Ts= 0.01

#N = 1000
#x = np.zeros((4,N+1))
#Initial state
#x[:,0] = x0
#Input
#u = 100*np.ones(N+1)

#Kp = 2.0
#for i in range(1,N):
#    x[:,i] = system.state_update(x[:,i-1],u[i-1],Ts)


#plt.plot(x[0,:])
#plt.show()









