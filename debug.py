from post_process import *
from time import time

initial_time = time()

# output = "../post_process/from_ATLAS2.0/N=900_h=1.0_rhoH=0.5_AF_square_ECMC"
output = "../post_process/from_ATLAS2.0/N=8100_h=1.0_rhoH=0.88_AF_square_ECMC"
psi23 = PsiMN(output, 2, 3)
psi23.calc_write(calc_correlation=False)
theta = np.angle(np.sum(psi23.op_vec))
# theta = -1.7225563044616816
pos_args = (output, theta, 0.4)
G = RealizationsAveragedOP(17, PositionalCorrelationFunction, pos_args)
print("Elapsed time: " + str(time() - initial_time))
