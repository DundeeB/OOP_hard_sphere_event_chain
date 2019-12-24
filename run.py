from EventChainActions import *
from SnapShot import View2D
import numpy as np
import os, shutil, random

# Input
rho_H = 0.85  # closest rho_H, the code will generate it with different but close value
h = 1
n_row = 50
n_col = 18
n_sp_per_dim_per_cell = 1

# More physical properties calculated from Input
N = n_row*n_col
N_iteration = 100*N
dn_save = N
equib_cycles = 4*dn_save
r = 1
sig = 2*r
H = (h+1)*sig

# build input parameters for cells
a_dest = sig*np.sqrt(2/(rho_H*(1+h)*np.sin(np.pi/3)))
l_y_dest = a_dest * n_row/2 * np.sin(np.pi/3)
e = n_sp_per_dim_per_cell*a_dest
n_col_cells = int(n_col/n_sp_per_dim_per_cell)
n_row_cells = int(round(l_y_dest/e))
l_x = n_col_cells * e
l_y = n_row_cells * e
a = np.sqrt(l_x*l_y/N)
rho_H = (sig**2)/((a**2)*(h+1))

# Folder Handeling
sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_triangle_ECMC'
output_dir = '../simulation-results/' + sim_name
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

# Simulation description
print("New rho_H chosen: " + str(rho_H))
print("N=" + str(N) + ", N_iterations=" + str(N_iteration) +
      ", Lx=" + str(l_x) + ", Ly=" + str(l_y))

# construct array of cells and fill with spheres
arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells)
arr.add_third_dimension_for_sphere(H)
arr.generate_spheres_in_AF_triangular_structure(n_row, n_col, r)
total_step = a * np.sqrt(n_row) * 0.1

# Initialize View
draw = View2D(output_dir, arr.boundaries)
draw.array_of_cells_snapshot('Initial Conditions', arr, 'Initial Conditions')
draw.dump_spheres(arr.all_centers, 'Initial Conditions')
draw.save_matlab_Input_parameters(arr.all_spheres[0].rad, rho_H)

# Run loops
for i in range(N_iteration):
    #Choose sphere
    while True:
        i_all_cells = random.randint(0, len(arr.all_cells) - 1)
        cell = arr.all_cells[i_all_cells]
        if len(cell.spheres) > 0:
            break
    i_sphere = random.randint(0, len(cell.spheres) - 1)
    sphere = cell.spheres[i_sphere]

    #Choose v_hat
    t = np.random.random() * np.pi
    if i % 2 == 0:
        v_hat = (np.sin(t), 0, np.cos(t))
    else:
        v_hat = (0, np.sin(t), np.cos(t))
    v_hat = np.array(v_hat)/np.linalg.norm(v_hat)

    #perform step
    step = Step(sphere, total_step, v_hat, arr.boundaries)
    i_cell, j_cell = cell.ind[:2]
    arr.perform_total_step(i_cell, j_cell, step)

    # save
    if (i+1) % dn_save == 0 and i+1 > equib_cycles:
        draw.dump_spheres(arr.all_centers, str(i + 1))
    if (i+1) % (N_iteration/100) == 0:
        print(str(100*(i+1) / N_iteration) + "%", end=", ")
