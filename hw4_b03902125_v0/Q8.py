import numpy as np 
import sys
from numpy.linalg import inv
import matplotlib.pyplot as plt

train = []
with open(sys.argv[1], 'r') as trainfile:
	for r in trainfile:
		r = r.split()
		r = list(map(lambda x:float(x), r))
		train.append(r)
train = np.array(train)

test = []
with open(sys.argv[2], 'r') as testfile:
	for r in testfile:
		r = r.split()
		r = list(map(lambda x:float(x), r))
		test.append(r)
test = np.array(test)

Ein_all = []
Eout_all = []
Eval_all = []

for i in range(2, -11, -1):

	X = np.hstack((np.ones((120, 1)), train[:120,:-1]))

	multiplier = 10**i
	wreg = np.matmul(np.matmul(inv(np.matmul(X.T, X)+multiplier*np.identity(np.matmul(X.T, X).shape[0])), X.T), train[:120,-1])

	y = np.sign(np.matmul(X, wreg))

	err01 = (y!=train[:120,-1])
	Ein = 0
	for e in err01:
		if e:
			Ein += 1

	print('Ein=', Ein/err01.shape[0])

	Ein_all.append(Ein/err01.shape[0])

	X = np.hstack((np.ones((80, 1)), train[120:,:-1]))
	y = np.sign(np.matmul(X, wreg))

	err01 = (y!=train[120:,-1])
	Eval = 0
	for e in err01:
		if e:
			Eval += 1

	print('Eval=', Eval/err01.shape[0])
	Eval_all.append(Eval/err01.shape[0])


	X = np.hstack((np.ones((test.shape[0], 1)), test[:,:-1]))
	y = np.sign(np.matmul(X, wreg))

	err01 = (y!=test[:,-1])
	Eout = 0
	for e in err01:
		if e:
			Eout += 1

	print('Eout=', Eout/err01.shape[0])
	Eout_all.append(Eout/err01.shape[0])

Ein_all = np.array(Ein_all)
Eout_all = np.array(Eout_all)
Eval_all = np.array(Eval_all)

#Eout_all = Eout_all[::-1]

min_idx = np.argmin(Ein_all)

#min_idx = Eout_all.shape[0]-1-min_idx

#Eout_all = Eout_all[::-1]
print('Ein_all[min_idx == %d] = %f'%(min_idx, Ein_all[min_idx]))
print('Eval_all[min_idx == %d] = %f'%(min_idx, Eval_all[min_idx]))
print('Eout_all[min_idx == %d] = %f'%(min_idx, Eout_all[min_idx]))

plt_Etrain, = plt.plot(np.arange(2, -11, -1), Ein_all)
plt_Eval, = plt.plot(np.arange(2, -11, -1), Eval_all)
plt.title('Plot of Etrain&Eval vs. logλ(base=10)')# give plot a title
plt.xlabel('logλ(base=10)')# make axis labels
plt.ylabel('Etrain&Eval')
plt.legend([plt_Etrain, plt_Eval], ('Etrain', 'Eval'))
plt.show()