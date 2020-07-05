from post_process import *
from time import time

initial_time = time()

# output = "../post_process/from_ATLAS2.0/N=900_h=1.0_rhoH=0.5_AF_square_ECMC"
output = "../post_process/from_ATLAS2.0/N=8100_h=1.0_rhoH=0.88_AF_square_ECMC"
psi23 = PsiMN(output, 2, 3)
psi23.calc_write(calc_correlation=False)
theta = np.angle(np.sum(psi23.op_vec))
# theta = -1.7225563044616816
G = PositionalCorrelationFunction(output, theta=theta, rect_width=0.2)
G.calc_write(bin_width=0.2)
print("Elapsed time: " + str(time() - initial_time))
