#!/Local/cmp/anaconda3/bin/python -u
import os
import time
import re
from send_parametric_runs import params_from_name

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results2.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def send_specific_run(sim_name, post_types):
    for post_type in post_types:
        out_pwd = prefix + 'out/post_process_' + sim_name + '_' + post_type + '.out'
        err_pwd = prefix + 'out/post_process_' + sim_name + '_' + post_type + '.err'
        time.sleep(2.0)
        os.system("qsub -V -v sim_path=" + sim_name.replace('=', '\=') + ",run_type=" + post_type +
                  " -N " + post_type + "_" + sim_name + " -o " + out_pwd + " -e " + err_pwd +
                  " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " + code_prefix + "post_process_env.sh")


def main():
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for sim_name in sims:
        N, h, rhoH, ic = params_from_name(sim_name)
        # ["psi23", "psi14", "psi16", "pos"]
        if h >= 1.0:
            send_specific_run(["psi23"])
        if h == 0.8:
            send_specific_run(["psi23"])
        if h <= 0.4:
            send_specific_run(["psi16"])


if __name__ == "__main__":
    main()
    # send_specific_run("N=8100_h=1.0_rhoH=0.86_AF_triangle_ECMC",["psi23"])
