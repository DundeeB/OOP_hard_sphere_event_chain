#!/Local/cmp/anaconda3/bin/python -u
import os
import time
# from send_parametric_runs import params_from_name
import re

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def send_specific_run(sim_name, post_types):
    for post_type in post_types:
        out_pwd = prefix + 'out/post_process_' + sim_name + '_' + post_type + '.out'
        err_pwd = prefix + 'out/post_process_' + sim_name + '_' + post_type + '.err'
        time.sleep(2.0)
        os.system("qsub -V -v sim_path=" + sim_name.replace('=', '\=') + ",run_type=" + post_type +
                  " -N " + post_type + "_" + sim_name + " -o " + out_pwd + " -e " + err_pwd +
                  " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " + code_prefix + "post_process_env.sh")


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
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for sim_name in sims:
        N, h, rhoH, ic = params_from_name(sim_name)
        if h >= 1.0:
            send_specific_run(sim_name, ["psi23", "Bragg_S23", "Bragg_Sm23", "pos23"])
        if h == 0.8:
            send_specific_run(sim_name, ["psi14", "burger_square", "Bragg_S14", "Bragg_Sm14", "pos14"])
        if h <= 0.4:
            send_specific_run(sim_name, ["psi16", "Bragg_S16", "Bragg_Sm16", "pos16"])


if __name__ == "__main__":
    main()
