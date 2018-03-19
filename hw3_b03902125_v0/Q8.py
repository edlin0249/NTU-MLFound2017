import sys
import numpy as np 
import matplotlib.pyplot as plt
import pylab as pl

train_dat = open(sys.argv[1], 'r')
test_dat = open(sys.argv[2], 'r')

train_X = []
train_Y = []
for r in train_dat:
	r = r.split()
	t = list(map(lambda x: float(x), r))
	train_X.append(t[:-1])
	train_Y.append(t[-1])

test_X = []
test_Y = []
for r in test_dat:
	r = r.split()
	t = list(map(lambda x: float(x), r))
	test_X.append(t[:-1])
	test_Y.append(t[-1])

lr = float(sys.argv[3])
T = 2000

train_X = np.array(train_X)
train_X = np.hstack((np.ones((train_X.shape[0], 1)), train_X))
train_Y = np.array(train_Y)
train_Y = np.reshape(train_Y, (train_Y.shape[0], 1))
W = np.zeros(train_X.shape[1])

test_X = np.array(test_X)
test_X = np.hstack((np.ones((test_X.shape[0], 1)), test_X))
test_Y = np.array(test_Y)
test_Y = np.reshape(test_Y, (test_Y.shape[0], 1))

Ein = []
Eout = []

for i in range(T):
	#print('training')
	S = np.dot(train_X, W)
	#print(S.shape)
	S = np.reshape(S, (S.shape[0], 1))
	#print(S.shape)
	#print(S)
	S = -train_Y*S
	#print(S)
	#print(S.shape)
	theta = 1/(1+np.exp(-S))
	#print(theta.shape)
	tmp = -train_Y*train_X
	#print(tmp.shape)
	theta = theta*tmp
	#print(theta.shape)
	err = np.sum(theta, axis=0)
	#print(err.shape)
	gradient = err/train_X.shape[0]
	#print(gradient.shape)
	W = W-lr*gradient
	#print(W.shape)
	#print(W)
	#print()
	S = np.dot(train_X, W)
	#print(S)
	#S = np.reshape(S, (S.shape[0], 1))
	y_hat = 1/(1+np.exp(-S))
	#print(y_hat)
	for i in range(len(y_hat)):
		if y_hat[i] > 0.5:
			y_hat[i] = 1.0
		else:
			y_hat[i] = -1.0
	#print(y_hat.shape)
	y_hat = np.reshape(y_hat, (y_hat.shape[0], 1))
	#print(y_hat.shape)
	isnotequal = y_hat != train_Y
	#print(isnotequal)
	err = 0
	for j in range(len(isnotequal)):
		if isnotequal[j][0]:
			err += 1

	err01_in = err/len(isnotequal)
	Ein.append(err01_in)
	#print('Ein = %f'%err01_in)
	#print('testing')
	S = np.dot(test_X, W)
	#print(S)
	#S = np.reshape(S, (S.shape[0], 1))
	y_hat = 1/(1+np.exp(-S))
	#print(y_hat)
	for i in range(len(y_hat)):
		if y_hat[i] > 0.5:
			y_hat[i] = 1.0
		else:
			y_hat[i] = -1.0
	#print(y_hat.shape)
	y_hat = np.reshape(y_hat, (y_hat.shape[0], 1))
	#print(y_hat.shape)
	isnotequal = y_hat != test_Y
	#print(isnotequal)
	err = 0
	for j in range(len(isnotequal)):
		if isnotequal[j][0]:
			err += 1
	err01_out = err/len(isnotequal)
	Eout.append(err01_out)

	#print('Eout = %f'%err01_out)

W = np.zeros(train_X.shape[1])
Ein_SGD = []
Eout_SGD = []

for i in range(T):
	#print('training')
	S = np.dot(train_X[i%train_X.shape[0]], W)
	#print(S.shape)
	S = -train_Y[i%train_Y.shape[0]]*S
	#print(S.shape)
	theta = 1/(1+np.exp(-S))
	#print(theta.shape)
	tmp = -train_Y[i%train_Y.shape[0]]*train_X[i%train_X.shape[0]]
	#print(tmp.shape)
	err = theta*tmp
	#print(err.shape)
	W = W-lr*err
	#print(W.shape)
	#print(W)
	#print()
	S = np.dot(train_X, W)
	#print(S)
	#S = np.reshape(S, (S.shape[0], 1))
	y_hat = 1/(1+np.exp(-S))
	#print(y_hat)
	for i in range(len(y_hat)):
		if y_hat[i] > 0.5:
			y_hat[i] = 1.0
		else:
			y_hat[i] = -1.0
	#print(y_hat.shape)
	y_hat = np.reshape(y_hat, (y_hat.shape[0], 1))
	#print(y_hat.shape)
	isnotequal = y_hat != train_Y
	#print(isnotequal)
	err = 0
	for j in range(len(isnotequal)):
		if isnotequal[j][0]:
			err += 1

	err01_in = err/len(isnotequal)
	Ein_SGD.append(err01_in)
	#print('Ein = %f'%err01_in)

	#print('testing')
	S = np.dot(test_X, W)
	#print(S)
	#S = np.reshape(S, (S.shape[0], 1))
	y_hat = 1/(1+np.exp(-S))
	#print(y_hat)
	for i in range(len(y_hat)):
		if y_hat[i] > 0.5:
			y_hat[i] = 1.0
		else:
			y_hat[i] = -1.0
	#print(y_hat.shape)
	y_hat = np.reshape(y_hat, (y_hat.shape[0], 1))
	#print(y_hat.shape)
	isnotequal = y_hat != test_Y
	#print(isnotequal)
	err = 0
	for j in range(len(isnotequal)):
		if isnotequal[j][0]:
			err += 1

	err01_out = err/len(isnotequal)
	Eout_SGD.append(err01_out)
	#print('Eout = %f'%err01_out)

#plot 
iteration = np.arange(1,T+1)
pl.plot(iteration, Ein, label='gradient descent')
pl.plot(iteration, Ein_SGD, label='stochastic gradient descent')
pl.title('Ein v.s. iterations, lr='+str(lr))# give plot a title
pl.xlabel('iterations')# make axis labels
pl.ylabel('Ein')
pl.legend()# make legend
pl.show()