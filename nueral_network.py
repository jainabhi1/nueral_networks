import numpy as np 


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


class nueral_net(object):

	def __init__(self,sizes):

		self.sizes = sizes
		self.layers = len(sizes)
		self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
		self.bias = [np.random.randn(y,1) for y in sizes[1:]]

	def train(self,x,y):

		for k in range(10000):
			l = len(x)

			for i in range(l):
				
				z = x[i]
				zs = [z]
			
				for w,b in zip(self.weights,self.bias):
					q = np.dot(w,z) + b
					z = sigmoid(q)
					zs.append(z)

				# if k == 9999:
				# 	print zs[-1],out[i]

				delta = zs[-1]*(1-zs[-1])*(-out[i]+zs[-1])
				deltas = [delta]

				for j in range(1,self.layers-1):
					delta = np.dot(self.weights[-j].transpose(), delta) * zs[-j-1] *(1-zs[-j-1])
					deltas.append(delta)
				
				for j in range(1,self.layers):
					self.weights[-j] = self.weights[-j] - deltas[j-1]*zs[-j-1].transpose()
					self.bias[-j] = self.bias[-j] - deltas[j-1]

	def predict(self,x):
		z = x
		for w,b in zip(self.weights,self.bias):
			q = np.dot(w,z) + b
			z = sigmoid(q)

		if z > 0.5:
			print 1
		else:
			print 0


net = nueral_net(np.array([2,2,1]))

a = np.array([0,1])
inp = []
out = []
for x in a:
	for y in a[::-1]:
		inp.append( np.array([x,y]).reshape((2,1)) )
		out.append(x^y);


net.train(inp,out)

net.predict(np.array([0,0]).reshape(2,1))