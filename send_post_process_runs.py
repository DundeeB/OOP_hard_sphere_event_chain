#!/Local/ph_daniel/anaconda3/bin/python -u
import os
import time
import re
import csv
from SnapShot import WriteOrLoad

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


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


def mn_from_sim(sim_name):
    _, h, _, _, _ = params_from_name(sim_name)
    if h > 0.85:
        mn = "23"
    if 0.55 <= h <= 0.85:
        mn = "14"
        # send_specific_run(sim_name, ["BurgersSquare"])
    if h < 0.55:
        mn = "16"
    return mn


def create_op_dir(sim):
    op_dir = os.path.join(prefix, sim, "OP")
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)
    return


def main():
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for sim in sims:
        create_op_dir(sim)
    f = open(os.path.join(code_prefix, 'post_process_list.txt'), 'wt')
    try:
        writer = csv.writer(f, lineterminator='\n')
        for sim_name in sims:
            N, h, rhoH, ic, algorithm = params_from_name(sim_name)
            if h != 0.8:
                continue
            if N == 30 ** 2 or (N == 200 ** 2 and ic == 'honeycomb') or (N == 300 ** 2 and ic == 'square'):
                for calc_type in ["Ising-annealing", "psi_mean"]:
                    writer.writerow((sim_name, calc_type + "14"))
            if 0.7 <= rhoH <= 0.9 and ic == 'square' and N == 300 ** 2:
                for calc_type in ["psi", "Bragg_S", "Bragg_Sm"]:
                    writer.writerow((sim_name, calc_type + "14"))
                    if rhoH in [0.77, 0.775, 0.78, 0.785, 0.79, 0.8, 0.81]:
                        writer.writerow((sim_name, "LocalPsi_radius=30_14"))
    finally:
        f.close()
        os.system("condor_submit post_process.sub")


if __name__ == "__main__":
    main()

# TODO: debug Local psi see ATLAS err file
