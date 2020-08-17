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
    rho_H_arr = [0.5, 0.6, 0.7, 0.75] + [round(r, 2) for r in np.linspace(0.8, 1, int((1 - 0.8) / 0.01) + 1)]
    # for h in [1, 0.8]:
    for h in [0.01, 0.1]:  # 2d runs
        for n_factor in [1, 3, 5, 11, 20]:
            for rho_H in rho_H_arr:
                n_row = 30 * n_factor
                n_col = 30 * n_factor
                send_single_run_envelope(h, n_row, n_col, rho_H, 'square')
                n_row = 50 * n_factor
                n_col = 18 * n_factor
                send_single_run_envelope(h, n_row, n_col, rho_H, 'honeycomb')

    # const eta=pi/4*(N/2)*sig^2/A=pi/8*H/sig*rhoH
    # for rho_H_h1 in [0.89, 0.9, 0.91]:
    #     h1 = 1
    #     eta = np.pi / 8 * (1 + h1) * rho_H_h1
    #     for h in [1.1, 1.2, 1.3, 1.4]:
    #         rho_H = 8 * eta / np.pi / (1 + h)

    # for rho_H in [0.775, 0.78, 0.785, 0.79, 0.795]:
    #   h = 1.4
    #     for n_factor in [1, 3, 5, 11]:
    #         n_row = 30 * n_factor
    #         n_col = 30 * n_factor
    #         send_single_run_envelope(h, n_row, n_col, rho_H, 'square')
    #         n_row = 50 * n_factor
    #         n_col = 18 * n_factor
    #         send_single_run_envelope(h, n_row, n_col, rho_H, 'honeycomb')


if __name__ == "__main__":
    main()
