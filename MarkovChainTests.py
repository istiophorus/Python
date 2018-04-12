import numpy as np


def matpow(mx,x):
    if x < 1:
        raise ValueError
        
    if x != int(x):
        raise ValueError
        
    if x == 1:
        return mx

    mxm = mx
    while x > 1:
        mxm = np.matmul(mxm, mx)
        x -= 1
        
    return mxm


def nstepmc(initial,mx,n):
    if n < 1:
        raise ValueError
        
    if n != int(n):
        raise ValueError

    return np.matmul(initial, matpow(mx,n))    

initial = np.matrix([[0.1,0.9]])
mx = np.matrix([[0.6,0.4],[0.15,0.85]])

res = nstepmc(initial,mx,150)

print(res)

# gambler
mxg = np.matrix([[1,0,0,0],[0.5,0,0.5,0],[0,0.5,0,0.5],[0,0,0,1]])
initialg = np.matrix([0,0,1,0])
res2 = nstepmc(initialg,mxg,100)

print(res)