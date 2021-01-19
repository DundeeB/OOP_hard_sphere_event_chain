#!/Local/ph_daniel/anaconda3/bin/python -u
import os
import numpy as np
import time
import re
import csv

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def sim_name(N, h, rhoH, ic):
    if ic == 'honeycomb':
        init_name_in_dir = 'AF_triangle_ECMC'
    else:
        if ic == 'square':
            init_name_in_dir = 'AF_square_ECMC'
        else:
            if ic == 'triangle':
                init_name_in_dir = "triangle_ECMC"
            else:
                raise NotImplementedError(
                    "Implemented initial conditions are: square, honeycomb. No " + ic + " implemented")
    return "N=" + str(N) + "_h=" + str(h) + "_rhoH=" + str(rhoH) + "_" + init_name_in_dir


def send_runs_envelope(sims_names):
    f = open(os.path.join(code_prefix, 'ecmc_list.txt'), 'wt')
    try:
        writer = csv.writer(f, lineterminator='\n')
        for sim_name in sims_names:
            writer.writerow([sim_name])
    finally:
        f.close()
        os.system("condor_submit ECMC.sub")
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


def main():
    runs = []

    # All runs - usefull for rerunning everything
    # for d in os.listdir(prefix):
    #     if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d)):
    #         runs.append(d)
    runs.append(sim_name(N=100 ** 2, h=0.8, rhoH=0.1, ic='square'))
    # for N in [100 ** 2, 200 ** 2, 300 ** 2]:
    #     Low density runs
    #     for h in [0.1, 0.6, 0.8, 1.0]:
    #         for rhoH in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #             if h >= 0.6:
    #                 runs.append(sim_name(N, h, rhoH, 'square'))
    #             if h >= 0.8:
    #                 runs.append(sim_name(N, h, rhoH, 'honeycomb'))
    #             if h == 0.1:
    #                 runs.append(sim_name(N, h, rhoH, 'triangle'))
    #
    #     Nominal h=0.8,1.0 runs
    #     rhoH_runs = {1.0: np.round(np.linspace(0.8, 0.9, 11), 2), 0.8: np.round(np.linspace(0.75, 0.85, 11), 2)}
    #     rhoH_runs = {1.0: [0.845, 0.855, 0.865], 0.8: [0.775, 0.785, 0.795]}
    #
    #     for h in [0.8, 1.0]:
    #         for rhoH in rhoH_runs[h]:
    #             runs.append(sim_name(N, h, rhoH, 'square'))
    #             runs.append(sim_name(N, h, rhoH, 'honeycomb'))
    #
    #     Following DOI: 10.1039/c4sm00125g, at h=0.1 eta*sig/H=pi/4*rhoH phase transition at 0.64-0.67, that is rhoH at
    #     0.81-0.85
    #         h = 0.1
    #         for rhoH in np.round(np.linspace(0.78, 0.88, 11), 2):
    #             runs.append(sim_name(N, h, rhoH, 'triangle'))
    #
    #         for h in [0.6]:
    #             for rhoH in np.round(np.linspace(0.73, 0.83, 11), 2):
    #                 runs.append(sim_name(N, h, rhoH, 'square'))

    send_runs_envelope(runs)


if __name__ == "__main__":
    main()
