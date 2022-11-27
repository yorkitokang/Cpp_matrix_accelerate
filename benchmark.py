import numpy as np;
import time
import torch
size = 32000
m1 = np.random.rand(size,size)
m2 = np.random.rand(size,size)
#time_start=time.time() 
#m3 = np.dot(m1,m2)
#time_end=time.time()
#print(f"Elapsed time during the {size} mul using numpy in seconds:",time_end-time_start)

t1 = torch.tensor(m1)
t2 = torch.tensor(m2)
time_start=time.time() 
torch.mm(t1,t2)
time_end=time.time()
print(f"Elapsed time during the {size} mul using numpy in seconds:",time_end-time_start)
