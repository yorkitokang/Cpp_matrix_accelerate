import numpy as np;
import time
import torch
size = 16000
m1 = np.random.rand(size,size)
m2 = np.random.rand(size,size)
# time_start=time.time() 
# m3 = np.dot(m1,m2)
# time_end=time.time()
# print(f"Elapsed time during the {size} mul using numpy in seconds:",time_end-time_start)

# t1 = torch.tensor(m1)
# t2 = torch.tensor(m2)
# time_start=time.time() 
# torch.mm(t1,t2)
# time_end=time.time()
# print(f"Elapsed time during the {size} mul using torch in seconds:",time_end-time_start)

t3 = torch.tensor(m1).cuda()
t4 = torch.tensor(m2).cuda()
time_start=time.time() 
torch.mm(t3,t4)
time_end=time.time()
print(f"Elapsed time during the {size} mul using torch cuda in seconds:",time_end-time_start)
