import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


MTTF1 = 10
MTTF2 = 20
MTTR1 = 2
MTTR2 = 3

l1 = 1/MTTF1
l2 = 1/MTTF2
m1 = 1/MTTR1
m2 = 1/MTTR2

Q = np.array([[-l1-l2, l1, l2,0], [m1, -m1-l2, 0, l2], [m2, 0, -m2-l1, l1], [0, m2, m1, -m2-m1]])

Pi0 = np.array([1.,0.,0.,0.])

def fun(t, Pi):
    return Pi @ Q

tMax = 10
t = np.linspace(0, tMax, 101)
    
res = integrate.solve_ivp(fun, (0, tMax), Pi0, t_eval=t)

plt.plot(res.t, res.y.T)
plt.show()