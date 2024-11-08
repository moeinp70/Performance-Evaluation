import numpy as np
import matplotlib.pyplot as plt
from collections import deque


tMax = 10

s = 1
t = 0

sxt = deque()
sxt.append([t, s])

ts1 = 0
ts2 = 0
ts3 = 0

RTT = deque()
curRTT = 0

while t < tMax:
    if s == 1:
        dt = 0.05 + np.random.rand() * 0.05
        ts1 = ts1 + dt
        curRTT = curRTT + dt
        ns = 2
    if s == 2:
        dt = 0.25 + np.random.rand() * 0.25
        ts2 = ts2 + dt
        curRTT = curRTT + dt
        ns = 3
    if s == 3:
        dt = 0.1 + np.random.rand() * 0.1
        ts3 = ts3 + dt
        curRTT = curRTT + dt
        RTT.append(curRTT)
        curRTT = 0
        ns = 1
    
    t = t + dt
    s = ns
    sxt.append([t, s])
    
#print(sxt)
sxtA = np.array(list(sxt))

plt.stairs(sxtA[0:-1,1], sxtA[:,0])
plt.show()

print("Prob. Task 1: ", ts1 / t)
print("Prob. Task 2: ", ts2 / t)
print("Prob. Task 3: ", ts3 / t)
print("Average time between two executions of the same task: ", np.mean(list(RTT)))