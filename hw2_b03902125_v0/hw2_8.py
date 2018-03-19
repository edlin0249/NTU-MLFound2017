import numpy as np 
import matplotlib.pyplot as plt
#import plotly.plotly as py

iterations = 1000
sizes = 20

Eout = np.zeros(iterations)
Ein = np.zeros(iterations)

for i in range(iterations):
	#generate a size of data
	x = np.random.uniform(-1, 1, sizes)
	x = np.sort(x)
	#generate corresponding labels for each data
	y = np.random.uniform(size=sizes) >= 0.8
	y = [-np.sign(x[j])if e else np.sign(x[j]) for j, e in enumerate(y)]


	x = np.concatenate((x, [1]), axis=0)
	x = np.concatenate(([-1], x), axis=0)
	
	theta = np.array([(x[j]+x[j+1])/2 for j in range(x.shape[0]-1)]) #get thet

	best_errin = sizes+1
	best_theta = sizes+1
	best_s = 0

	for j in range(sizes+1):   # iterate over theta
		tmp = np.sign(x[1:sizes+1] - theta[j])

		###s = 1 case
		s = 1.0
		h = s*tmp

		comp = h != y
		tmp_err = 0.0
		for e in comp:
			if e:
				tmp_err += 1.0

		if tmp_err < best_errin:
			best_errin = tmp_err
			best_theta = theta[j]
			best_s = 1.0

		###s = -1 case
		s = -1.0
		h = s*tmp

		comp = h != y
		tmp_err = 0.0
		for e in comp:
			if e:
				tmp_err += 1.0

		if tmp_err < best_errin:
			best_errin = tmp_err
			best_theta = theta[j]
			best_s = -1.0

	Eout[i] = 0.5 + 0.3*best_s*(abs(best_theta)-1)
	Ein[i] = best_errin / float(sizes)

Eout_avg = np.mean(Eout)
Ein_avg = np.mean(Ein)

print("Eout_avg = %f, Ein_arg = %f" %(Eout_avg, Ein_avg))

plt.scatter(Ein, Eout)
plt.show()