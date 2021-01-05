#!/bin/bash
qsub -V -N sending_processing_runs -o out/send_process.out -e out/send_process.err -l nodes=1:ppn=1,mem=1gb,vmem=2gb -q S ./send_post_process_runs.py
