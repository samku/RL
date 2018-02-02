import sys
sys.path.append(r"casadiinstalldir")
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

dx = np.zeros((4,1))
dx[1,0]=1.
dx[2,0]=2.
print dx[2,0]