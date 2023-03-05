import numpy as np
import scipy.io as sio

folder = 'results/'

C_file = folder + 'C.txt'
B_file = folder + 'B.txt'
epsilon_file = folder + 'epsilon.txt'

N = 100000
D = 128

M = 2
K = 256

C = np.zeros([M,D,K])
B = np.zeros([N,M])
epsilon = 0




with open(C_file, 'r') as f:
	C_string = f.read()

C_split = C_string.split(' ')

count = 0
for ii in range(M):
	for jj in range(D):
		for kk in range(K):
			C[ii,jj,kk] = float(C_split[count])
			count = count + 1




with open(B_file, 'r') as f:
	B_string = f.read()

B_split = B_string.split(' ')

count = 0
for ii in range(N):
	for jj in range(M):
		B[ii,jj] = int(B_split[count])
		count = count + 1





with open(epsilon_file, 'r') as f:
	epsilon_string = f.read()

epsilon = float(epsilon_string)



sio.savemat('results.mat', {'C':C, 'B':B, 'epsilon': epsilon})












