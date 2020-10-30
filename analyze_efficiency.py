from run_functions import *
import matplotlib.pyplot as plt

# TODO: estimate ATLAS performence
# t, d, r = [], [], []
# for iterations in range(200, 1000, 200):
iterations = 500
# initial_time = time.time()
realizations, displacements = run_honeycomb(0.8, 10000, 0.8, record_displacements=True, iterations=iterations,
                                            write=False)
# t.append(time.time() - initial_time)
# d.append(displacements[-1])
# r.append(realizations[-1])
# print("\nDisplacements per sec are: " + str(d[-1] / t[-1]) + "\n")

plt.figure()
size = 30
params = {'legend.fontsize': 'large',
          'figure.figsize': (20, 8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
          'axes.titlepad': 25,
          'font.weight': 'bold'}
plt.rcParams.update(params)

# plt.plot(t, d, 'o', label='data from simulations')
# p = np.polyfit(t, d, 1)
# plt.plot(t, np.polyval(p, t), '--k', label='displacement per sec = ' + str(int(np.round(p[0], -2))), linewidth=3)
# plt.legend()
# plt.xlabel('time')
# plt.ylabel('displacements')
# plt.grid()
# plt.xlim([0, np.max(t) * 1.2])

# plt.subplot(212)
plt.plot(realizations, displacements, 'o', label='data from simulations')
p = np.polyfit(realizations, displacements, 1)
plt.plot(realizations, np.polyval(p, realizations), '--k', label='displacement per realization = ' + str(int(np.round(p[0], -2))), linewidth=3)
plt.legend()
plt.xlabel('realization')
plt.ylabel('displacements')
plt.grid()

plt.show()
