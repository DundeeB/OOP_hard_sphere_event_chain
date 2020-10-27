import subprocess
import sys

post_process_path = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\post_process\\from_ATLAS2.0\\N=8100_h=0.8_rhoH=0.81_AF_triangle_ECMC'
process_type = 'psi14mean'
subprocess.call([sys.executable, './post_process.py', post_process_path, process_type])
