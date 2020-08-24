#!/Local/cmp/anaconda3/bin/python -u
import os
import time
import re

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results2.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def main():
    sims = [d for d in os.listdir(prefix) if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for d in sims:
        if not re.match('.*h=1.0.*', d):
            continue
        for run_type in ["psi23"]:  # "psi14", "psi16", "pos"
            out_pwd = prefix + 'out/post_process_' + d + '_' + run_type + '.out'
            err_pwd = prefix + 'out/post_process_' + d + '_' + run_type + '.err'

            time.sleep(2.0)
            os.system(
                "qsub -V -v sim_path=" + d.replace('=', '\=') + ",run_type=" + run_type +
                " -N " + run_type + "_" + d + " -o " + out_pwd + " -e " + err_pwd +
                " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " + code_prefix + "post_process_env.sh")


if __name__ == "__main__":
    main()
