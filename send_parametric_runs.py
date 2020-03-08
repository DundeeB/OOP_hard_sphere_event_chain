#!/Local/cmp/anaconda3/bin/python -u
import os
import numpy as np
import time


def send_single_run_envelope(h, n_row, n_col, rhoH, initial_conditions):
    params = "h=" + str(h) + ",n_row=" + str(n_row) + ",n_col=" + str(n_col) + ",rhoH=" + str(rhoH) + \
             ",initial_conditions=" + initial_conditions
    if initial_conditions == 'honeycomb':
        init_name_in_dir = 'AF_triangle_ECMC'
    else:
        if initial_conditions == 'square':
            init_name_in_dir = 'AF_square_ECMC'
        else:
            raise NotImplementedError("Implemented initial conditions are: square, honeycomb")
    sim_name = "N=" + str(n_row*n_col) + "_h=" + str(h) + "_rhoH=" + str(rhoH) + \
             "_" + init_name_in_dir
    out_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + sim_name + ".out"
    err_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + sim_name + ".err"
    os.system("qsub -V -v " + params + " -N " + sim_name + " -o " + out_pwd + " -e " + err_pwd +
              " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " +
              "/srv01/technion/danielab/ECMC/OOP_hard_sphere_event_chain/py_env.sh")
    time.sleep(2.0)


rho_H_arr = [round(x, 2) for x in np.linspace(0.5, 1.0, 11)] + [0.68, 0.73, 0.88, 0.93, 0.98]
for h in [1, 0.8]:  # [0.7]:
    for n_factor in [1, 2, 3]:  # [2, 3]:
        for rho_H in rho_H_arr:
            n_row = 30 * n_factor
            n_col = 30 * n_factor
            send_single_run_envelope(h, n_row, n_col, rho_H, 'square')
            n_row = 50 * n_factor
            n_col = 18 * n_factor
            send_single_run_envelope(h, n_row, n_col, rho_H, 'honeycomb')
