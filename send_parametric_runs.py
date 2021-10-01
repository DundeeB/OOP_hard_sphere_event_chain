#!/Local/ph_daniel/anaconda3/bin/python -u
import os
import numpy as np
import time
import re
import csv

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def sim_name(N, h, rhoH, ic, algorithm):
    if ic == 'honeycomb':
        init_name_in_dir = 'AF_triangle_'
    else:
        if ic == 'square':
            init_name_in_dir = 'AF_square_'
        else:
            if ic == 'triangle':
                init_name_in_dir = "triangle_"
            else:
                raise NotImplementedError(
                    "Implemented initial conditions are: square, honeycomb. No " + ic + " implemented")
    return "N=" + str(N) + "_h=" + str(h) + "_rhoH=" + str(rhoH) + "_" + init_name_in_dir + algorithm


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
    algorithm = ss[-1]
    return N, h, rhoH, ic, algorithm


def main():
    runs = []
    N, rhoH, h = 30**2, 0.8, 0.8
    for ic in ['square', 'honeycomb']:
        for algorithm in ['ECMC', 'MCMC']:
            runs.append(sim_name(N, h, rhoH, ic, algorithm))
    send_runs_envelope(runs)


if __name__ == "__main__":
    main()
