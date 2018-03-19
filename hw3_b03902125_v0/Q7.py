import numpy as np 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

Ein = []
Eout = []
for i in range(1000):
	X = np.random.uniform(-1, 1, (1000, 2))
	y = np.sign(X[:,0]**2+X[:,1]**2-0.6)
	idx = np.random.randint(0, 1000, 100)  #flip 10% to make the noise
	y[idx] = -y[idx]
	#transform X
	x1_2 = X[:,0]**2
	x1_2 = np.reshape(x1_2, (1000, 1))
	x1x2 = X[:,0]*X[:,1]
	x1x2 = np.reshape(x1x2, (1000, 1))
	x2_2 = X[:,1]**2
	x2_2 = np.reshape(x2_2, (1000, 1))

	X = np.hstack((np.ones((1000, 1)), X, x1x2, x1_2, x2_2))

	#print(X.shape)
	W = np.dot(np.linalg.pinv(X), y)
	#print(W)
	y_hat = np.sign(np.dot(X, W))
	isnotequal = y_hat != y
	err = 0
	for j in range(len(isnotequal)):
		if isnotequal[j]:
			err += 1

	Ein.append(err/len(isnotequal))

#test
	X = np.random.uniform(-1, 1, (1000, 2))
	y = np.sign(X[:,0]**2+X[:,1]**2-0.6)
	idx = np.random.randint(0, 1000, 100)  #flip 10% to make the noise
	y[idx] = -y[idx]
	#transform X
	x1_2 = X[:,0]**2
	x1_2 = np.reshape(x1_2, (1000, 1))
	x1x2 = X[:,0]*X[:,1]
	x1x2 = np.reshape(x1x2, (1000, 1))
	x2_2 = X[:,1]**2
	x2_2 = np.reshape(x2_2, (1000, 1))

	X = np.hstack((np.ones((1000, 1)), X, x1x2, x1_2, x2_2))

	y_hat = np.sign(np.dot(X, W))
	isnotequal = y_hat != y
	err = 0
	for j in range(len(isnotequal)):
		if isnotequal[j]:
			err += 1

	Eout.append(err/len(isnotequal))


print('avg. Ein is %f'%np.mean(Ein))
print('avg. Eout is %f'%np.mean(Eout))
iteration = np.arange(1,1000+1)
plt.hist(Eout)
#plt.title('Eout v.s. iterations')# give plot a title
plt.xlabel('Eout')# make axis labels
#plt.ylabel('Eout')
#plt.legend()# make legend
plt.show()