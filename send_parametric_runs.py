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
    sim_name = "N=" + str(n_row * n_col) + "_h=" + str(h) + "_rhoH=" + str(rhoH) + \
               "_" + init_name_in_dir
    out_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + sim_name + ".out"
    err_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + sim_name + ".err"
    os.system("qsub -V -v " + params + " -N " + sim_name + " -o " + out_pwd + " -e " + err_pwd +
              " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " +
              "/srv01/technion/danielab/OOP_hard_sphere_event_chain/py_env.sh")
    time.sleep(2.0)


def quench_single_run_envelope(action, other_sim_dir, desired_rho_or_h):
    params = "action=" + action + ",other_sim_dir=" + other_sim_dir + ",desired_rho_or_h=" + str(desired_rho_or_h)
    sim_name = action + "_" + other_sim_dir + "_desired_rho_or_h=" + str(desired_rho_or_h)
    out_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + sim_name + ".out"
    err_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + sim_name + ".err"
    os.system("qsub -V -v " + params + " -N " + sim_name + " -o " + out_pwd + " -e " + err_pwd +
              " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " +
              "/srv01/technion/danielab/OOP_hard_sphere_event_chain/py_quench_env.sh")
    time.sleep(2.0)


desired_h = 0.8
for initial_rho in [round(r,2) for r in np.linspace(0.75, 0.85, 11)]:
    sim_for_quench = 'N=8100_h=1.0_rhoH=' + str(initial_rho) + '_AF_triangle_ECMC'
    action = 'zquench'
    quench_single_run_envelope(action, sim_for_quench, desired_rho_or_h=desired_h)
quench_single_run_envelope('zquench', ' N=900_h=1.0_rhoH=0.75_AF_triangle_ECMC', 0.8)

rho_H_arr = [0.7, 0.81, 0.82, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
for h in [1, 0.8]:  # , 0.7]:
    for n_factor in [3, 4]:  # [1, 2, 3]:
        for rho_H in rho_H_arr:
            n_row = 30 * n_factor
            n_col = 30 * n_factor
            send_single_run_envelope(h, n_row, n_col, rho_H, 'square')
            n_row = 50 * n_factor
            n_col = 18 * n_factor
            send_single_run_envelope(h, n_row, n_col, rho_H, 'honeycomb')
send_single_run_envelope(h, 60, 60, 0.88, 'square')
send_single_run_envelope(h, 100, 36, 0.88, 'honeycomb')
