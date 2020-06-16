#!/Local/cmp/anaconda3/bin/python -u
import os
import numpy as np
import time

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results2.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


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
    out_pwd = prefix + "out/" + sim_name + ".out"
    err_pwd = prefix + "out/" + sim_name + ".err"
    time.sleep(2.0)
    os.system("qsub -V -v " + params + " -N " + sim_name + " -o " + out_pwd + " -e " + err_pwd +
              " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " + code_prefix + "py_env.sh")
    return


def quench_single_run_envelope(action, other_sim_dir, desired_rho_or_h):
    params = "action=" + action + ",other_sim_dir=" + other_sim_dir + ",desired_rho_or_h=" + str(desired_rho_or_h)
    sim_name = action + "_" + other_sim_dir + "_desired_rho_or_h=" + str(desired_rho_or_h)
    out_pwd = prefix + "out/" + sim_name + ".out"
    err_pwd = prefix + "out/" + sim_name + ".err"
    time.sleep(2.0)
    os.system("qsub -V -v " + params + " -N " + sim_name + " -o " + out_pwd + " -e " + err_pwd +
              " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " + code_prefix + "py_quench_env.sh")
    return


def main():
    # desired_h = 0.8
    # for initial_rho in [round(r,2) for r in np.linspace(0.75, 0.85, 11)]:
    #     sim_for_quench = 'N=8100_h=1.0_rhoH=' + str(initial_rho) + '_AF_triangle_ECMC'
    #     action = 'zquench'
    #     quench_single_run_envelope(action, sim_for_quench, desired_rho_or_h=desired_h)
    # quench_single_run_envelope('zquench', ' N=900_h=1.0_rhoH=0.75_AF_triangle_ECMC', 0.8)
    #
    # rho_H_arr = [0.5, 0.6, 0.7] + [round(r, 2) for r in np.linspace(0.75, 1, (1 - 0.75) / 0.01 + 1)]
    # for h in [1, 0.8]:
    #     for n_factor in [1, 2, 3, 4]:
    #         for rho_H in rho_H_arr:
    #             n_row = 30 * n_factor
    #             n_col = 30 * n_factor
    #             send_single_run_envelope(h, n_row, n_col, rho_H, 'square')
    #             n_row = 50 * n_factor
    #             n_col = 18 * n_factor
    #             send_single_run_envelope(h, n_row, n_col, rho_H, 'honeycomb')

    rho_H_arr = [0.5]
    for h in [1]:
        for n_factor in [1]:
            for rho_H in rho_H_arr:
                n_row = 30 * n_factor
                n_col = 30 * n_factor
                send_single_run_envelope(h, n_row, n_col, rho_H, 'square')
                n_row = 50 * n_factor
                n_col = 18 * n_factor
                send_single_run_envelope(h, n_row, n_col, rho_H, 'honeycomb')

if __name__ == "__main__":
    main()