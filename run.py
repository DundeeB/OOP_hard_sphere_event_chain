from EventChainActions import *
from SnapShot import View2D
import numpy as np
import os, random

# Input
# closest rho_H, the code will generate it with different but close value
for h in [1, 0.8]:
    for n_factor in [2, 1]:
        for rho_H in [0.6, 0.7, 0.85, 0.95]:
            n_row = n_factor*50
            n_col = n_factor*18
            n_sp_per_dim_per_cell = 1

            # More physical properties calculated from Input
            N = n_row*n_col
            r = 1
            sig = 2*r
            H = (h+1)*sig

            # Numeric choices calibrated for fast convergence
            N_iteration = int(N*2e3)
            dn_save = N
            equib_cycles = N * 50

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

            # Simulation description
            print("\n\nSimulation: N=" + str(N) + ", rhoH=" + str(rho_H) +", h=" + str(h))
            print("N_iterations=" + str(N_iteration) +
                  ", Lx=" + str(l_x) + ", Ly=" + str(l_y))

            # construct array of cells and fill with spheres
            arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells)
            arr.add_third_dimension_for_sphere(H)
            total_step = a * np.sqrt(n_row) * 0.05

            # Initialize View and folder, and add spheres
            sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_triangle_ECMC'
            output_dir = '../simulation-results/' + sim_name
            draw = View2D(output_dir, arr.boundaries)
            if os.path.exists(output_dir):
                last_centers, last_ind = draw.last_spheres()
                sp = [Sphere(tuple(c), r) for c in last_centers]
                arr.append_sphere(sp)
                print("Simulation with same parameters exist already, continuing from last file")
            else:
                os.mkdir(output_dir)
                arr.generate_spheres_in_AF_triangular_structure(n_row, n_col, r)
                draw.array_of_cells_snapshot('Initial Conditions', arr, 'Initial Conditions')
                draw.dump_spheres(arr.all_centers, 'Initial Conditions')
                draw.save_matlab_Input_parameters(arr.all_spheres[0].rad, rho_H)
                last_ind = 0  # count starts from 1 so 0 means non exist yet and the first one will be i+1=1

            # Run loops
            for i in range(last_ind, N_iteration):
                #Choose sphere
                while True:
                    i_all_cells = random.randint(0, len(arr.all_cells) - 1)
                    cell = arr.all_cells[i_all_cells]
                    if len(cell.spheres) > 0:
                        break
                i_sphere = random.randint(0, len(cell.spheres) - 1)
                sphere = cell.spheres[i_sphere]

                #Choose v_hat
                t = np.random.random() * np.pi  # TBD try converging faster with t = (0.5-np.random.random()) * np.arccos(H/total_step)
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
                if i+1 == equib_cycles:
                    print("\nFinish equilibrating")
