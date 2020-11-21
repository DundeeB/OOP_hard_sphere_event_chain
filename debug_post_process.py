from post_process import *
import matplotlib.pyplot as plt
from time import time

t0 = time()
post_process_path = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS2.0\\N=900_h=0.8_rhoH=0.81_AF_triangle_ECMC'
psi_op = PsiMN(post_process_path, 1, 4)
psi_op.calc_order_parameter(calc_upper_lower=False)
bragg = BraggStructure(post_process_path, psi_op)
bragg.calc_peak(randomize=False)
t1 = time()
print('Bragg elapsed time: ' + str(t1 - t0))
braggM = MagneticBraggStructure(post_process_path, psi_op)
braggM.calc_peak(randomize=False)
t2 = time()
print('Magnetic Bragg elapsed time: ' + str(t2 - t1))
# for real in np.power(10, range(2, 8)):
#     bragg.S(bragg.k_perf(), realizations=int(real))
#
# reals = [d[3] for d in bragg.data]
# S_values = [np.abs(d[2]) for d in bragg.data]
#
# size = 30
# params = {'legend.fontsize': 'large',
#           'figure.figsize': (20, 8),
#           'axes.labelsize': size,
#           'axes.titlesize': size,
#           'xtick.labelsize': size * 0.75,
#           'ytick.labelsize': size * 0.75,
#           'axes.titlepad': 25,
#           'font.weight': 'bold'}
# plt.rcParams.update(params)
# plt.grid()
#
# plt.semilogx(reals, S_values, '--o', label='N=8100 Randomized realizations out of N^2 couples', linewidth=1.5)
# correct_S = bragg.S(bragg.k_perf(), randomize=False)
# plt.semilogx(len(bragg.spheres) ** 2, correct_S, 'ok', label='converged value for all N^2 couples', markersize=15)
# plt.xlabel('Couples realizations')
# plt.ylabel('|S| value at $k_{perf}$')
# plt.legend()
# plt.show()
