#!/Local/cmp/anaconda3/bin/python

import os


def send_single_run_envelope(h, n_row, n_col, rho_H, initial_conditions):
    params = "h=" + str(h) + ",n_row=" + str(n_row) + ",n_col=" + str(n_row) + ",rho_H=" + str(rho_H) + \
             ",initial_conditions=" + initial_conditions
    out_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + params + ".out"
    err_pwd = "/storage/ph_daniel/danielab/ECMC_simulation_results/out/" + params + ".err"
    os.system("qsub -V -v " + params + " -o " + out_pwd + " -e " + err_pwd +
              " -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q N " +
              "/srv01/technion/danielab/ECMC/OOP_hard_sphere_event_chain/py_env.sh")


send_single_run_envelope(1, 30, 30, 0.6, 'square')
# for h in [1, 0.8]:
#     for n_factor, rho_H in  zip([1, 2]):
