from post_process import Ising
import numpy as np
import matplotlib.pyplot as plt

sim_dir = '../post_process/from_ATLAS3.0/N=10000_h=0.8_rhoH=0.8_AF_square_ECMC'
ising_instance = Ising(sim_dir, k_nearest_neighbors=4)  # loads last_cv_spins, which is not at criticality but still...
Jc = -0.52
initial_iterations = int(2e3 * ising_instance.N)
cv_iterations = int(1e4 * ising_instance.N)
ising_instance.initialize(J=Jc)
ising_instance.anneal(initial_iterations, diter_save=initial_iterations)
_, E_vec, _ = ising_instance.anneal(cv_iterations, diter_save=ising_instance.N)

Cvs = []
for bin_size in range(1, int(len(E_vec) / 2)):
    Cvs.append(np.mean([np.var(E_vec[k::bin_size], ddof=1) / ising_instance.N for k in range(bin_size)]))
plt.plot(Cvs, '.-')
