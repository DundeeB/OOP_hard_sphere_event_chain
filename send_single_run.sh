#!/bin/bash

qsub -V -v h=1,n_row=30,n_col=30,rho_H=0.6,initial_conditions=square -o /storage/ph_daniel/danielab/outmsg -e /storage/ph_daniel/danielab/outerr -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q S py_env.sh