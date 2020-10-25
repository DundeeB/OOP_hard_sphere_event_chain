from run_functions import *
import matplotlib.pyplot as plt

t, d = [], []
for iterations in range(200, 1000, 200):
    initial_time = time.time()
    realizations, displacements = run_honeycomb(0.8, 30, 30, 0.79, record_displacements=True, iterations=iterations,
                                             write=False)
    t.append(time.time() - initial_time)
    d.append(displacements[-1])
    print("\nDisplacements per sec are: " + str(d[-1] / t[-1]) + "\n")

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

plt.plot(t, d, 'o', label='data from simulations')
p = np.polyfit(t, d, 1)
plt.plot(t, np.polyval(p, t), '--k', label='displacement per sec = ' + str(int(np.round(p[0], -2))), linewidth=3)
plt.legend()
plt.xlabel('time')
plt.ylabel('displacements')
plt.grid()
plt.xlim([0, np.max(t) * 1.2])
plt.show()
