#!/bin/bash
qsub -V -N sending_runs -o out/send.out -e out/send.err -l nodes=1:ppn=1,mem=1gb,vmem=4gb -q S ./send_parametric_runs.py
