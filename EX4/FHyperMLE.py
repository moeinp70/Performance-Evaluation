import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt

#ax = plt.axes(projection='3d')

col = 1
trc = "Trace.csv"

if len(sys.argv) > 1:
    trc = sys.argv[1]
if len(sys.argv) > 2:
    col = int(sys.argv[2])

trace = np.loadtxt(trc, delimiter=";")
N = trace.shape[0]
probV = np.r_[1.:N+1] / N

srt = trace.copy()
srt.sort(0)

M1t = np.sum(srt[:,col]) / N
M2t = np.sum(srt[:,col]**2) / N
M3t = np.sum(srt[:,col]**3) / N

def fun(x):
    l1 = x[0]
    l2 = x[1]
    p1 = x[2]
    p2 = 1-p1
    
    return -np.sum(np.log(p1*l1*np.exp(-l1*srt[:,col]) + p2*l2*np.exp(-l2*srt[:,col])))




sx = opt.minimize(fun, np.array([0.8/M1t,1.2/M1t,0.4]), bounds=((0.001, 100.0), (0.001, 100.0), (0.001, 0.999)), constraints=[{'type': 'ineq', 'fun': lambda x:  x[1] - x[0] - 0.001}])
l1d = sx.x[0]
l2d = sx.x[1]
p1d = sx.x[2]
p2d = 1 - p1d

print("p1, l1, l2 = ", p1d, l1d, l2d)

M1d = p1d / l1d + p2d / l2d
M2d = 2*(p1d / l1d**2 + p2d / l2d**2)
M3d = 6*(p1d / l1d**3 + p2d / l2d**3)


t = np.r_[1.:1001] / 50

Fhypo = 1 - p1d * np.exp(-t * l1d) - p2d * np.exp(-t * l2d)

plt.plot(srt[:,col], probV, ".")
plt.plot(t, Fhypo)
plt.show()


print("First  moment: ", M1d, M1t)
print("Second moment: ", M2d, M2t)
print("Third  moment: ", M3d, M3t)
