import numpy as np 
import sys 
import time 
_,n_dim, n_iter = sys.argv
n_dim = int(n_dim)
n_iter = int(n_iter)
warmup = int(n_iter/2)
A = np.ones(n_dim,dtype=np.float32)
B = np.ones(n_dim,dtype=np.float32)
# for a simple loop

mean = 0
for i in range(n_iter):
    start = time.monotonic_ns()
    _ = np.dot( A, B)
    end = time.monotonic_ns()
    if i > warmup :
        mean = mean + (end - start)

mean = mean/warmup / 1e9 # ns > s 
flops = 2*n_dim / mean /1e9 # flops > GFLOP/s
bandwidth = 8*n_dim / mean / 1e9 # bytes > GB/s

print(f"N: {n_dim}\tT: {mean} sec\tB: {bandwidth} GB/sec\tF: {flops} GFLOP/s\n");