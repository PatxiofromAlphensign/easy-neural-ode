from datasets import miniboone
import jax
import numpy as np

def algo_jax(x):
	#assert isinstance(x, jax)
	if len(x.shape) < 2:
		x = x.reshape(x.shape[0], 1)
	if x.shape[0] < x.shape[1]**2:
		raise ValueError("dim 0 should be a scaler mulitple of dim 1")
	for i in range(1, x.shape[0]):
		for k in  range(1, x.shape[1]):
			x = jax.ops.index_update(x , jax.ops.index[:i*k,k], np.ones(i*k))
	return x			

def algo(x):
	if len(x.shape) < 2:
		x = x.reshape(x.shape[0], 1)
	for i in range(x.shape[0]):
		for k in  range(x.shape[1]):
			x[k:i,:] = np.arange(k, i).reshape(k-i, 1) 
	return x			


class data_d:
	def __init__(self, n, ar=None):
		self.data = np.array(range(n))
		self.data_x = algo(self.data)
		self.m0 = miniboone.MINIBOONE.Data(self.data)
		self.m1 = miniboone.MINIBOONE.Data(self.data_x)
	def avg_count(self):
		i=0
		for k in self.m0.x:
			i += k/self.m0.N
		self.i = i	
		return i


if __name__=="__main__":
	d = data_d(20)
	data = algo_jax(jax.numpy.arange(8).reshape(4,2)) #dim 0 should be a scaler mulitple of dim 1
	print(jjax())


