import os
import time

prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results2.0/"
code_prefix = "/srv01/technion/danielab/OOP_hard_sphere_event_chain/"


def main():
    sims = [d for d in os.listdir('prefix') if d.startswith('N=') and os.path.isdir(os.path.join(prefix, d))]
    for d in sims:
        out_pwd = prefix + 'out/post_process_' + d + '.out'
        err_pwd = prefix + 'out/post_process_' + d + '.err'
        time.sleep(0.5)
        os.chdir(os.path.join(prefix, d))
        os.system(
            "qsub -V -N post_process_" + d + " -o " + out_pwd + " -e " + err_pwd +
            " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " + code_prefix + "post_process_env.sh")


if __name__ == "__main__":
    main()
