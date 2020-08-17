#!/bin/bash
/Local/cmp/anaconda3/bin/python -u /srv01/technion/danielab/OOP_hard_sphere_event_chain/post_process.py $sim_path psi23
sleep 2.0
/Local/cmp/anaconda3/bin/python -u /srv01/technion/danielab/OOP_hard_sphere_event_chain/post_process.py $sim_path psi14
sleep 2.0
/Local/cmp/anaconda3/bin/python -u /srv01/technion/danielab/OOP_hard_sphere_event_chain/post_process.py $sim_path psi16
sleep 2.0
/Local/cmp/anaconda3/bin/python -u /srv01/technion/danielab/OOP_hard_sphere_event_chain/post_process.py $sim_path pos