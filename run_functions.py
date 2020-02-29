#!/Local/cmp/anaconda3/bin/python

from EventChainActions import *
from SnapShot import View2D
import numpy as np
import os, random, sys


def run_sim(initial_arr, N, h, rho_H, total_step, sim_name):
    # Numeric choices calibrated for fast convergence
    N_iteration = int(N * 2e3)
    dn_save = N
    equib_cycles = N * 50

    # Simulation description
    print("\n\nSimulation: N=" + str(N) + ", rhoH=" + str(rho_H) + ", h=" + str(h))
    print("N_iterations=" + str(N_iteration) +
          ", Lx=" + str(initial_arr.l_x) + ", Ly=" + str(initial_arr.l_y))

    # Initialize View and folder, and add spheres
    code_dir = os.getcwd()
    output_dir = '/storage/ph_daniel/danielab/ECMC_simulation_results/' + sim_name
    # output_dir = r'C:\Users\Daniel Abutbul\OneDrive - Technion\simulation-results\\' + sim_name
    draw = View2D(output_dir, initial_arr.boundaries)
    if os.path.exists(output_dir):
        last_centers, last_ind = draw.last_spheres()
        sp = [Sphere(tuple(c), initial_arr.all_spheres[0].rad) for c in last_centers]
        # construct array of cells and fill with spheres
        arr = Event2DCells(edge=initial_arr.edge, n_rows=initial_arr.n_rows, n_columns=initial_arr.n_columns)
        arr.add_third_dimension_for_sphere(initial_arr.l_z)
        arr.append_sphere(sp)
        print("Simulation with same parameters exist already, continuing from last file")
    else:
        os.mkdir(output_dir)
        arr = initial_arr
        draw.array_of_cells_snapshot('Initial Conditions', arr, 'Initial Conditions')
        draw.dump_spheres(arr.all_centers, 'Initial Conditions')
        draw.save_matlab_Input_parameters(arr.all_spheres[0].rad, rho_H)
        last_ind = 0  # count starts from 1 so 0 means non exist yet and the first one will be i+1=1
    os.chdir(output_dir)

    # Run loops
    for i in range(last_ind, N_iteration):
        # Choose sphere
        while True:
            i_all_cells = random.randint(0, len(arr.all_cells) - 1)
            cell = arr.all_cells[i_all_cells]
            if len(cell.spheres) > 0:
                break
        i_sphere = random.randint(0, len(cell.spheres) - 1)
        sphere = cell.spheres[i_sphere]

        # Choose v_hat
        t = np.random.random() * np.pi
        # TBD try converging faster with t = (0.5-np.random.random()) * np.arccos(H/total_step)
        if i % 2 == 0:
            v_hat = (np.sin(t), 0, np.cos(t))
        else:
            v_hat = (0, np.sin(t), np.cos(t))
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)

        # perform step
        step = Step(sphere, total_step, v_hat, arr.boundaries)
        i_cell, j_cell = cell.ind[:2]
        arr.perform_total_step(i_cell, j_cell, step)

        # save
        if (i + 1) % dn_save == 0 and i + 1 > equib_cycles:
            draw.dump_spheres(arr.all_centers, str(i + 1))
        if (i + 1) % (N_iteration / 100) == 0:
            print(str(100 * (i + 1) / N_iteration) + "%", end=", ")
        if i + 1 == equib_cycles:
            print("\nFinish equilibrating")
    os.system('echo \'Finished ' + str(N_iteration) + ' iterations\' > FINAL_MESSAGE')
    os.chdir(code_dir)
    return 0


def run_honeycomb(h, n_row, n_col, rho_H):
    # More physical properties calculated from Input
    N = n_row * n_col
    r = 1
    sig = 2 * r

    # build input parameters for cells
    n_sp_per_dim_per_cell = 1
    a_dest = sig * np.sqrt(2 / (rho_H * (1 + h) * np.sin(np.pi / 3)))
    l_y_dest = a_dest * n_row / 2 * np.sin(np.pi / 3)
    e = n_sp_per_dim_per_cell * a_dest
    n_col_cells = int(n_col / n_sp_per_dim_per_cell)
    n_row_cells = int(round(l_y_dest / e))
    l_x = n_col_cells * e
    l_y = n_row_cells * e
    a = np.sqrt(l_x * l_y / N)
    rho_H = (sig ** 2) / ((a ** 2) * (h + 1))
    total_step = a * np.sqrt(n_row) * 0.05

    initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells)
    initial_arr.add_third_dimension_for_sphere((h+1)*sig)
    initial_arr.generate_spheres_in_AF_triangular_structure(n_row, n_col, r)
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_triangle_ECMC'
    return run_sim(initial_arr, N, h, rho_H, total_step, sim_name)


def run_square(h, n_row, n_col, rho_H):
    # More physical properties calculated from Input
    N = n_row * n_col
    r = 1
    sig = 2 * r
    H = (h + 1) * sig

    # build input parameters for cells
    n_sp_per_dim_per_cell = 2
    a = sig * np.sqrt(1 / (rho_H * (1 + h)))
    e = n_sp_per_dim_per_cell * a
    n_col_cells = int(n_col / n_sp_per_dim_per_cell)
    n_row_cells = int(n_row / n_sp_per_dim_per_cell)
    l_x = n_col_cells * e
    l_y = n_row_cells * e
    a_free = ((l_x * l_y * H - 4 * np.pi / 3 * (r ** 3)) / N) ** (1 / 3)
    total_step = a_free * np.sqrt(n_row) / 10  # avoid maximal depth recursion error by division over 10

    # construct array of cells and fill with spheres
    initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells)
    initial_arr.add_third_dimension_for_sphere(H)
    initial_arr.generate_spheres_in_AF_square(n_row, n_col, r)

    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_square_ECMC'
    return run_sim(initial_arr, N, h, rho_H, total_step, sim_name)

args = sys.argv[1:]

if len(args) != 5:
    raise ValueError("Wrong number of argmuents. Should give h, n_row, n_col, rho_H, inital conditions")
h, n_row, n_col, rho_H = [float(x) for x in args[:-1]]
n_col = int(n_col)
n_row = int(n_row)
if args[-1] == 'square':
    run_square(h, n_row, n_col, rho_H)
    # print("run_square(" + str(h) + ", " + str(n_row) + ", " + str(n_col) + ", " + str(rho_H) + ")")
else:
    if args[-1] == 'honeycomb':
        run_honeycomb(h, n_row, n_col, rho_H)
        # print("run_honeycomb(" + str(h) + ", " + str(n_row) + ", " + str(n_col) + ", " + str(rho_H) + ")")
    else:
        raise NotImplemented("Implemented initial conditions are: square, honeycomb")
