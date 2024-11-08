import numpy as np
import matplotlib.pyplot as plt

srv = 1.0
arr = 0.8

N = 1000
K0 = 50
DeltaK = 10
MaxK = 1000

maxRelErr = 0.01

U1 = 0
U2 = 0

K = K0
Krange = K

while K < MaxK:
    for k in range(0, Krange):
        u = np.random.rand(N,1)
        Xsrv = - np.log(1 - u) / srv

        u = np.random.rand(N,1)
        Xarr = - np.log(1 - u) / arr

        A = np.zeros((N, 1))
        C = np.zeros((N, 1))

        A[0,0] = Xarr[0,0]
        C[0,0] = Xarr[0,0] + Xsrv[0,0]

        for i in range(1, N):
            A[i,0] = A[i-1,0] + Xarr[i,0]
            C[i,0] = max(A[i,0], C[i-1,0]) + Xsrv[i,0]

        T = C[N-1,0]
        B = np.sum(Xsrv)
        Uk = B / T

        U1 = U1 + Uk
        U2 = U2 + Uk*Uk

    EU     = U1 / K
    EU2    = U2 / K
    VarU   = EU2 - EU*EU
    SigmaU = np.sqrt(VarU)
    DeltaU  = 1.96 * np.sqrt(VarU / K)
    Ul = EU - DeltaU
    Uu = EU + DeltaU
    RelErrU = 2 * (Uu - Ul) / (Uu + Ul)

    if RelErrU < maxRelErr:
        break

    K = K + DeltaK
    Krange = DeltaK

print("E[U]     = ", EU)
print("E[U^2]   = ", EU2)
print("Var[U]   = ", VarU)
print("Sigma[U] = ", SigmaU)


print("95% confidence interval of U: ", Ul, Uu)
print("Relative error of U:          ", RelErrU)

print("Solution obtained in ", K, " iterations")