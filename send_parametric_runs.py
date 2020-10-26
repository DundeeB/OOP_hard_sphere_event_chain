#!/Local/cmp/anaconda3/bin/python -u
import os
import numpy as np
import time
import re

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def send_single_run_envelope(h, N, rhoH, initial_conditions):
    params = "N=" + str(N) + ",h=" + str(h) + ",rhoH=" + str(rhoH) + ",initial_conditions=" + initial_conditions
    if initial_conditions == 'honeycomb':
        init_name_in_dir = 'AF_triangle_ECMC'
    else:
        if initial_conditions == 'square':
            init_name_in_dir = 'AF_square_ECMC'
        else:
            raise NotImplementedError(
                "Implemented initial conditions are: square, honeycomb. No " + initial_conditions + " implemented")
    sim_name = "N=" + str(N) + "_h=" + str(h) + "_rhoH=" + str(rhoH) + "_" + init_name_in_dir
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


def params_from_name(name):
    ss = re.split("[_=]", name)
    for i, s in enumerate(ss):
        if s == 'N':
            N = int(ss[i + 1])
        if s == 'h':
            h = float(ss[i + 1])
        if s == 'rhoH':
            rhoH = float(ss[i + 1])
        if s == 'AF':
            ic = ss[i + 1]
    return N, h, rhoH, ic


def resend_all_runs():
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for d in sims:
        N, h, rhoH, ic = params_from_name(d)
        if ic == 'triangle':
            ic = 'honeycomb'
        send_single_run_envelope(h, int(np.sqrt(N)), int(np.sqrt(N)), rhoH, ic)
        # n_row and n_col are used only when a new simulation is sent, otherwise only N=n_row*n_col is calculated,
        # and the rest is read from the input file. Therefor, even if n_row and n_col are wrong here, the simulation
        # will continue from last restart with the correct values.


def main():
    # resend_all_runs()
    send_single_run_envelope(0.8, 100**2, 0.7, 'honeycomb')
    send_single_run_envelope(0.8, 100**2, 0.7, 'square')
    send_single_run_envelope(1.0, 100 ** 2, 0.75, 'honeycomb')
    send_single_run_envelope(1.0, 100 ** 2, 0.75, 'square')

    for N in [100 ** 2, 200 ** 2, 300 ** 2]:
        for h in [0.8, 1.0]:
            for rhoH in np.linspace(0.75, 0.85, 11) if h == 0.8 else np.linspace(0.8, 0.9, 11):
                send_single_run_envelope(h, N, rhoH, 'square')
                send_single_run_envelope(h, N, rhoH, 'honeycomb')


if __name__ == "__main__":
    main()
