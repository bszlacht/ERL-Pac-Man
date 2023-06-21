import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('stats_q_agent_medium_4.pkl.pkl', 'rb') as file:
    data = pickle.load(file)

y_values = [point[1] for point in data]

means = []
for i in range(0, len(y_values), 1000):
    mean = np.mean(y_values[i:i+1000])
    means.append(mean)

# Plot the means
x_values = np.arange(0, len(means))
plt.plot(x_values, means, 'o')
plt.xlabel('Interval')
plt.ylabel('Mean')
plt.title('Means of Every 1000th Point')
plt.show()