from post_process import *
import matplotlib.pyplot as plt
from time import time

t0 = time()
post_process_path = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS2.0\\N=900_h=0.8_rhoH=0.81_AF_triangle_ECMC'
psi_op = PsiMN(post_process_path, 1, 4)
psi_op.calc_order_parameter()
bragg = BraggStructure(post_process_path, psi_op)
bragg.calc_order_parameter()
t1 = time()
print('Bragg elapsed time: ' + str(t1 - t0))
braggM = MagneticBraggStructure(post_process_path, psi_op)
braggM.calc_order_parameter()
t2 = time()
print('Magnetic Bragg elapsed time: ' + str(t2 - t1))
bragg.correlation(realizations=int(1e5))
t3 = time()
print('Bragg Correlations elapsed time: ' + str(t3 - t2))
braggM.correlation(realizations=int(1e5))
t4 = time()
print('Magnetic Bragg Correlations elapsed time: ' + str(t4 - t3))

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

# plt.semilogx(reals, S_values, '--o', label='N=8100 Randomized realizations out of N^2 couples', linewidth=1.5)
for data, lbl in zip([bragg.data, braggM.data], ['S', 'S_m']):
    ks = [np.sqrt(d[0] ** 2 + d[1] ** 2) for d in data]
    S_values = [np.abs(d[2]) for d in data]
    plt.plot(ks, S_values, 'o', linewidth=1.5, label=lbl)
# correct_S = bragg.S(bragg.k_perf(), randomize=False)
# plt.semilogx(len(bragg.spheres) ** 2, correct_S, 'ok', label='converged value for all N^2 couples', markersize=15)
# plt.xlabel('Couples realizations')
plt.xlabel('|k|')
plt.legend()
plt.show()

plt.figure()
plt.loglog(bragg.corr_centers, bragg.op_corr, '.-', label='e^ikr correlation for k_peak of S(k)')
plt.loglog(braggM.corr_centers, braggM.op_corr, '.-', label='z*e^ikr correlation for k_peak of $S_M$(k)')
plt.legend()
plt.xlabel('$\\Delta$r')
plt.show()
