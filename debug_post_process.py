from post_process import *
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits import mplot3d

# TODO: figure out why calculation time doesnt scale as O(N) worst (per realization should scale as O(1))
t0 = time()
post_process_path = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS3.0\\N=10000_h=0.8_rhoH=0.79_AF_square_ECMC'
# post_process_path = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS2.0\\N=900_h=1.0_rhoH=0.5_AF_square_ECMC'
psi_op = PsiMN(post_process_path, 1, 4)
psi_op.calc_order_parameter()
t1 = time()
print('Psi elapsed time: ' + str(t1 - t0))
bragg = BraggStructure(post_process_path, psi_op)
bragg.calc_order_parameter()
t2 = time()
print('Bragg elapsed time: ' + str(t2 - t1))
braggM = MagneticBraggStructure(post_process_path, psi_op)
braggM.calc_order_parameter()
t2 = time()
print('Magnetic Bragg elapsed time: ' + str(t2 - t1))

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
plt.grid()
plt.figure(1)
fig2 = plt.figure(2)
for data, lbl, sub in zip([braggM.data, bragg.data], ['$S_m$', '$S$'], [1, 2]):
    plt.figure(1)
    plt.subplot(211)
    ks = [np.sqrt(d[0] ** 2 + d[1] ** 2) for d in data]
    S_values = [d[2] for d in data]
    plt.plot(ks, S_values, 'o', linewidth=1.5, label=lbl)
    plt.xlabel('|k|')
    plt.legend()
    plt.subplot(212)
    thetas = [np.arctan2(d[1], d[0]) for d in data]
    plt.plot(thetas, S_values, 'o', linewidth=1.5, label=lbl)
    plt.xlabel('$\\theta$')
    plt.legend()

    ax = fig2.add_subplot(2, 1, sub, projection='3d')
    ax.scatter([d[0] for d in data], [d[1] for d in data], S_values, '.')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_zlabel(lbl)
# plt.show()

reals = int(1e4)
t3 = time()
bragg.correlation(realizations=reals, randomize=True)
t4 = time()
print('Bragg Correlations pairs per sec: ' + str(reals / (t4 - t3)))
braggM.correlation(realizations=reals, randomize=True)
t5 = time()
print('Magnetic Bragg Correlations pairs per sec: ' + str(reals / (t5 - t4)))

plt.figure()
plt.loglog(bragg.corr_centers, np.abs(bragg.op_corr), '.-', label='e^ikr correlation for k_peak of S(k)')
plt.loglog(braggM.corr_centers, np.abs(braggM.op_corr), '.-', label='z*e^ikr correlation for k_peak of $S_M$(k)')
plt.legend()
plt.xlabel('$\\Delta$r')
plt.show()
