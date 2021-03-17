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
        # send_specific_run(sim_name, ["burger_square"])
    if h < 0.55:
        mn = "16"
    return mn


def create_op_dir(sim):
    op_dir = os.path.join(prefix, sim, "OP")
    if not os.path.exists(op_dir):
        os.mkdir(op_dir)
    return


def main():
    # sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    # sims = ["N=10000_h=0.8_rhoH=0.8_AF_square_ECMC"]

    # for sim in sims:
    #     create_op_dir(sim)
    # default_op = ["Ising-E_T"]
    # "psi", "Bragg_S", "Bragg_Sm", "pos", "gM", "Ising-annealing", "Ising-E_T", "Density","LocalPsi_radius=50_",
    # "LargestComponent",
    # ["LocalPsi_radius=" + str(r) + "_" for r in [20, 25, 35, 40]]
    f = open(os.path.join(code_prefix, 'post_process_list.txt'), 'wt')
    try:
        writer = csv.writer(f, lineterminator='\n')
        # for sim_name in sims:
        #     mn = mn_from_sim(sim_name)
        #     op_w_mn_list = [op + mn for op in default_op]
        #     # if mn == "14":
        #     #     op_w_mn_list += ["burger_square"]
        #     for calc_type in op_w_mn_list:
        #         writer.writerow((sim_name, calc_type))
        sims = ["N=90000_h=0.8_rhoH=" + str(rhoH) + "_AF_square_ECMC" for rhoH in [0.75, 0.8, 0.85]]
        for sim_name in sims:
            for real_count in [0, 1, 2, 3, 4]:
                load_obj = WriteOrLoad(os.path.join(prefix, sim_name))
                real = load_obj.realizations()[real_count]
                calc_type = 'Ising-E_T_real=' + str(real) + '_14'
                writer.writerow((sim_name, calc_type))
            writer.writerow((sim_name, "burger_square"))
            writer.writerow((sim_name, "Bragg_Sm14"))
    finally:
        f.close()
        os.system("condor_submit post_process.sub")


if __name__ == "__main__":
    main()
# TODO: decrease #files. For example, control when frustration_* is writen, because I am using it only for k=*_undirected
#  graph. Another example - I don't need local-psi_rad=10 for every burger calculation, its easy to calculate it and I
#  can recalculate it when ever. So is k=4_directed_graph
