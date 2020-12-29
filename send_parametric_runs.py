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
            if initial_conditions == 'triangle':
                init_name_in_dir = "triangle_ECMC"
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
        if s == 'triangle' or s == 'square':
            ic = s
            if ss[i - 1] == 'AF' and s == 'triangle':
                ic = 'honeycomb'
    return N, h, rhoH, ic


def resend_all_runs():
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for d in sims:
        N, h, rhoH, ic = params_from_name(d)
        send_single_run_envelope(h, N, rhoH, ic)


def main():
    # resend_all_runs()

    # send_single_run_envelope(0.8, 100 ** 2, 0.7, 'honeycomb')
    # send_single_run_envelope(0.8, 100 ** 2, 0.7, 'square')
    # send_single_run_envelope(1.0, 100 ** 2, 0.75, 'honeycomb')
    # send_single_run_envelope(1.0, 100 ** 2, 0.75, 'square')

    # for N in [100 ** 2, 200 ** 2, 300 ** 2]:
    #     for h in [0.8]:  # , 1.0]:
                    # for rhoH in np.round(np.linspace(0.75, 0.85, 11) if h == 0.8 else np.linspace(0.8, 0.9, 11), 2):
                    # for rhoH in [0.775, 0.785, 0.795] if h == 0.8 else [0.845, 0.855, 0.865]:
            # for rhoH in [0.86, 0.87, 0.88, 0.89, 0.90]:
            #     send_single_run_envelope(h, N, rhoH, 'square')
            #     send_single_run_envelope(h, N, rhoH, 'honeycomb')

    # for N in [100 ** 2, 200 ** 2, 300 ** 2]:
    #     h = 0.1
    # Following DOI: 10.1039/c4sm00125g, at h=0.1 eta*sig/H=pi/4*rhoH phase transition at 0.64-0.67, that is rhoH at
    # 0.81-0.85
    # for rhoH in np.round(np.linspace(0.78, 0.88, 11), 2):
    #     send_single_run_envelope(h, N, rhoH, 'triangle')

    for N in [100 ** 2, 200 ** 2, 300 ** 2]:
        for h in [0.6]:
            for rhoH in np.round(np.linspace(0.73, 0.83, 11), 2):
                send_single_run_envelope(h, N, rhoH, 'square')
                send_single_run_envelope(h, N, rhoH, 'triangle')


if __name__ == "__main__":
    main()
