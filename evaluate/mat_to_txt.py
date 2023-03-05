import numpy as np
import scipy.io as sio

sift1m = sio.loadmat('sift1M.mat')

base_in = sift1m['base']

base = np.transpose(base_in)
print base.shape


'''
with open('sift1m_base.txt', 'w') as f:
	for i in range(base.shape[0]):
		for j in range(base.shape[1]):
			f.write(str(int(base[i,j])))
			f.write(' ')
		f.write('\n')
'''


