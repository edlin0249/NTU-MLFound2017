import numpy as np
import random
import matplotlib.pyplot as plt

hw1_8_train = open("hw1_8_train.dat", "r")

training_set = []
train_label_set = []

for r in hw1_8_train:
	t = r.split()
	t = list(map(lambda x: float(x), t))
	training_set.append(t[:len(t)-1])
	train_label_set.append(t[len(t)-1])

hw1_8_train.close()

training_set = np.array(training_set)
#print(training_set)
#print(training_set.shape)
training_set = np.concatenate((np.ones((training_set.shape[0],1)), training_set), axis=1)
#print(training_set)
train_label_set = np.array(train_label_set)


index = list(range(training_set.shape[0]))
total = 0
updates_num = []
for _ in range(2000):
	w = np.zeros(training_set.shape[1])
	#iteration = 0
	updates = 0
	no_err = True
	n_idx = index[:]
	random.shuffle(n_idx)
	while no_err:
		#iteration += 1
		cnt = 0
		for i in n_idx:
			t = np.dot(w, training_set[i]);
			if t == 0:
				t = -1

			if np.sign(t) != np.sign(train_label_set[i]):
				w += train_label_set[i] * training_set[i]
				updates += 1
			else:
				cnt += 1

			if cnt == training_set.shape[0]:
				no_err = False

	updates_num.append(updates)
	total += updates

print(total/2000)

plt.hist(updates_num)
plt.xlabel("Number of Updates")
plt.ylabel("Frequency")
plt.show()

