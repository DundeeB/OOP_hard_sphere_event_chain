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
    # sims = ["N=40000_h=0.8_rhoH=0.81_AF_triangle_ECMC"]
    for sim in sims:
        create_op_dir(sim)
    f = open(os.path.join(code_prefix, 'post_process_list.txt'), 'wt')
    try:
        writer = csv.writer(f, lineterminator='\n')
        for sim_name in sims:
            N, h, rhoH, ic, algorithm = params_from_name(sim_name)
            # if N == 30 ** 2:
            #     writer.writerow((sim_name, "psi_mean14"))
            if h == 0.8 and rhoH in [0.77, 0.775, 0.78, 0.785, 0.8, 0.81]:  # 0.7 <= rhoH <= 0.9:
                # writer.writerow((sim_name, "psi_mean14"))
                if ic == 'square' and N == 300 ** 2:
                    for calc_type in ["LocalPsi_radius=30"]:  # ["psi", "Bragg_S", "Bragg_Sm", "Ising-annealing"]:
                        # + ["LocalPsi_radius=" + str(rad) + "_" for rad in [10, 30, 50]]
                        writer.writerow((sim_name, calc_type + "14"))
    finally:
        f.close()
        os.system("condor_submit post_process.sub")


if __name__ == "__main__":
    main()
