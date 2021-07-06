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
    return N, h, rhoH, ic


def mn_from_sim(sim_name):
    _, h, _, _ = params_from_name(sim_name)
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
    # sims = ["N=90000_h=0.8_rhoH=0.8_AF_square_ECMC"]
    for sim in sims:
        create_op_dir(sim)
    f = open(os.path.join(code_prefix, 'post_process_list.txt'), 'wt')
    try:
        writer = csv.writer(f, lineterminator='\n')

        for sim_name in sims:
            N, h, rhoH, _ = params_from_name(sim_name)
            if h == 0.8:
                # if N == 90000 and 0.75 <= rhoH <= 0.85:
                #     for rad in [10, 5, 2, 0]:
                #         writer.writerow((sim_name, "BurgersSquare_radius=" + str(rad)))
                writer.writerow((sim_name, "BurgersSquare"))
    finally:
        f.close()
        os.system("condor_submit post_process.sub")


if __name__ == "__main__":
    main()
