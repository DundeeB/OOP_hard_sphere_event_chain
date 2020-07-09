from post_process import *
from time import time

initial_time = time()

# output = "../post_process/from_ATLAS2.0/N=900_h=1.0_rhoH=0.5_AF_square_ECMC"
output = "../post_process/from_ATLAS2.0/N=8100_h=1.0_rhoH=0.88_AF_square_ECMC"
# psi23 = PsiMN(output, 2, 3)
# psi23.calc_write(calc_correlation=True)
# theta = np.angle(np.sum(psi23.op_vec))
# theta = -1.7225563044616816
# Pos = PositionalCorrelationFunction(output, theta, rect_width=0.2)
# Pos.calc_write(bin_width=0.2, write_vec=False, calc_upper_lower=True)
# pos_args = (output, theta, 1)
# G = RealizationsAveragedOP(3, PositionalCorrelationFunction, pos_args)
# G.calc_write(bin_width=0.2)

psi_args = (output, 2, 3)
P = RealizationsAveragedOP(2, PsiMN, psi_args)
P.calc_write(bin_width=0.2)
print("Elapsed time: " + str(time() - initial_time))
