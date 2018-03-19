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

for i in range(2, -11, -1):

	X = np.hstack((np.ones((train.shape[0], 1)), train[:,:-1]))

	multiplier = 10**i
	wreg = np.matmul(np.matmul(inv(np.matmul(X.T, X)+multiplier*np.identity(np.matmul(X.T, X).shape[0])), X.T), train[:,-1])

	y = np.sign(np.matmul(X, wreg))

	err01 = (y!=train[:,-1])
	Ein = 0
	for e in err01:
		if e:
			Ein += 1

	print('Ein=', Ein/err01.shape[0])

	Ein_all.append(Ein/err01.shape[0])

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

#Ein_all = Ein_all[::-1]

min_idx = np.argmin(Ein_all)

#min_idx = Ein_all.shape[0]-1-min_idx

#Ein_all = Ein_all[::-1]
print('Ein_all[min_idx == %d] = %f'%(min_idx, Ein_all[min_idx]))
print('Eout_all[min_idx == %d] = %f'%(min_idx, Eout_all[min_idx]))

plt_Ein, = plt.plot(np.arange(2, -11, -1), Ein_all)
plt_Eout, = plt.plot(np.arange(2, -11, -1), Eout_all)
plt.title('Plot of Ein&Eout vs. logλ(base=10)')# give plot a title
plt.xlabel('logλ(base=10)')# make axis labels
plt.ylabel('Ein&Eout')
#plt.xticks(np.arange(2, -11, -1))

plt.legend([plt_Ein, plt_Eout], ('Ein', 'Eout'))
plt.show()
