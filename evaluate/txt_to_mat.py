import numpy as np
import scipy.io as sio

N = 1000000
M = 2

B_for_base = np.zeros([N, M])

with open('B_for_base.txt', 'r') as f:
	B_for_base_str = f.read()

B_for_base_arr = B_for_base_str.split(' ')

count = 0
for i in range(N):
	for j in range(M):
		B_for_base[i,j] = int( B_for_base_arr[count] )
		count += 1

# sift1m = sio.loadmat('sift1M.mat')

# base_in = sift1m['base']

# base = np.transpose(base_in)
# print base.shape

a = {}
a['B_for_base'] = B_for_base
sio.savemat('B_for_base.mat', a)

