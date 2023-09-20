import numpy as np 
import sys 
import time 
_,n_dim, n_iter = sys.argv
n_dim = int(n_dim)
n_iter = int(n_iter)
A = np.ones(n_dim,dtype=np.float32)
B = np.ones(n_dim,dtype=np.float32)
# for a simple loop
def dp(N,A,B):
    R = 0.0;
    for j in range(0,N):
       R += A[j]*B[j]
    return R
mean = 0
for i in range(n_iter):
   start = time.monotonic_ns()
   _ = dp(n_dim, A, B)
   end = time.monotonic_ns()
   mean = mean + (end - start)
mean = mean/n_iter
print(f"Average runtime: {mean}")
   
