#!/Local/ph_daniel/anaconda3/bin/python -u
import os
import sys
from EventChainActions import *
from SnapShot import WriteOrLoad
import re
import time
from send_parametric_runs import *

epsilon = 1e-8
prefix = '/storage/ph_daniel/danielab/ECMC_simulation_results3.0/'


# prefix = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\simulation-results'


def run_honeycomb(h, N, rho_H, algorithm, **kwargs):
    # More physical properties calculated from Input
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_triangle_' + algorithm
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, h, rho_H, algorithm, sim_name, **kwargs)
        # when continuing from restart there shouldn't be use of initial arr
    else:
        n_row = int(np.sqrt(N))
        n_col = n_row
        r = 1
        sig = 2 * r
        # build input parameters for cells
        a_dest = sig * np.sqrt(2 / (rho_H * (1 + h) * np.sin(np.pi / 3)))
        l_y_dest = a_dest * n_row / 2 * np.sin(np.pi / 3)
        e = a_dest
        n_col_cells = n_col
        n_row_cells = int(round(l_y_dest / e))
        l_x = n_col_cells * e
        l_y = n_row_cells * e
        a = np.sqrt(l_x * l_y / N)
        rho_H_new = (sig ** 2) / ((a ** 2) * (h + 1))

        initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells, l_z=(h + 1) * sig)
        initial_arr.generate_spheres_in_AF_triangular_structure(n_row, n_col, r)
        initial_arr.scale_xy(np.sqrt(rho_H_new / rho_H))
        assert initial_arr.edge > sig

        return run_sim(initial_arr, N, h, rho_H, sim_name, **kwargs)


def run_square(h, N, rho_H, algorithm, **kwargs):
    # More physical properties calculated from Input
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_square_' + algorithm
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, h, rho_H, algorithm, sim_name, **kwargs)
        # when continuing from restart there shouldn't be use of initial arr
    else:
        n_row = int(np.sqrt(N))
        n_col = n_row  # Square initial condition for n_row!=n_col is not implemented...
        r, sig = 1.0, 2.0
        A = N * sig ** 2 / (rho_H * (1 + h))
        a = np.sqrt(A / N)
        n_row_cells, n_col_cells = int(np.sqrt(A) / (a * np.sqrt(2))), int(np.sqrt(A) / (a * np.sqrt(2)))
        e = np.sqrt(A / (n_row_cells * n_col_cells))
        assert e > sig, "Edge of cell is: " + str(e) + ", which is smaller than sigma."
        initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells, l_z=(h + 1) * sig)
        initial_arr.generate_spheres_in_AF_square(n_row, n_col, r)

        return run_sim(initial_arr, N, h, rho_H, sim_name, **kwargs)


def run_triangle(h, N, rho_H, algorithm, **kwargs):
    # More physical properties calculated from Input
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_triangle_' + algorithm
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, h, rho_H, algorithm, sim_name, **kwargs)
        # when continuing from restart there shouldn't be use of initial arr
    else:
        n_row = int(np.sqrt(N))
        n_col = n_row
        r, sig = 1.0, 2.0
        A = N * sig ** 2 / (rho_H * (1 + h))
        a = np.sqrt(A / N)
        n_row_cells, n_col_cells = int(np.sqrt(A) / (a * np.sqrt(2))), int(np.sqrt(A) / (a * np.sqrt(2)))
        e = np.sqrt(A / (n_row_cells * n_col_cells))
        assert e > sig, "Edge of cell is: " + str(e) + ", which is smaller than sigma."
        initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells, l_z=(h + 1) * sig)
        spheres = ArrayOfCells.spheres_in_triangular(n_row, n_col, r, initial_arr.l_x, initial_arr.l_y)
        initial_arr.append_sphere(spheres)
        initial_arr.update_all_spheres()
        assert initial_arr.legal_configuration()
        return run_sim(initial_arr, N, h, rho_H, algorithm, sim_name, **kwargs)


def run_sim(initial_arr, N, h, rho_H, algorithm, sim_name, iterations=None, record_displacements=False, write=True):
    if iterations is None:
        iterations = int(N * 1e4)
    rad = 1
    a_free = (1 / rho_H - np.pi / 6) ** (1 / 3) * 2 * rad  # ((V-N*4/3*pi*r^3)/N)^(1/3)
    if algorithm == 'ECMC':
        xy_total_step = a_free * np.sqrt(N)
        z_total_step = h * (2 * rad) * np.pi / 15  # irrational for the spheres to cover most of the z options
    elif algorithm == 'MCMC':
        metropolis_step = a_free * (np.pi / 3) / 10
    # Initialize View and folder, add spheres
    code_dir = os.getcwd()
    output_dir = os.path.join(prefix, sim_name)
    batch = os.path.join(output_dir, 'batch')
    if os.path.exists(output_dir) and write:
        files_interface = WriteOrLoad(output_dir, np.nan)
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = files_interface.load_Input()
        boundaries = [l_x, l_y, l_z]
        files_interface.boundaries = boundaries
        last_centers, last_ind = files_interface.last_spheres()
        sp = [Sphere(tuple(c), rad) for c in last_centers]
        # construct array of cells and fill with spheres
        arr = Event2DCells(edge=edge, n_rows=n_row, n_columns=n_col, l_z=l_z)
        arr.append_sphere(sp)
        arr.update_all_spheres()
        sys.stdout = open(batch, "a")
        print("\n-----------\nSimulation with same parameters exist already, continuing from last file.\n",
              file=sys.stdout)
    else:
        arr = initial_arr
        if write:
            files_interface = WriteOrLoad(output_dir, initial_arr.boundaries)
            os.mkdir(output_dir)
            sys.stdout = open(batch, "a")
            files_interface.dump_spheres(arr.all_centers, 'Initial Conditions')
            files_interface.save_Input(arr.all_spheres[0].rad, rho_H, arr.edge, arr.n_rows, arr.n_columns)
            os.chdir(output_dir)
        # print simulation description
        print("\n\nSimulation: N=" + str(N) + ", rhoH=" + str(rho_H) + ", h=" + str(h), file=sys.stdout)
        print("N_iterations=" + str(iterations) +
              ", Lx=" + str(initial_arr.l_x) + ", Ly=" + str(initial_arr.l_y), file=sys.stdout)
        last_ind = 0  # count starts from 1 so 0 means non exist yet and the first one will be i+1=1

    # Run loops
    day = 86400  # seconds
    hour = 3600  # seconds
    i = last_ind
    if i >= iterations:
        sys.exit(0)
    if record_displacements:
        displacements = [0]
        realizations = [i]
    initial_time = time.time()
    realizations_saved = 0
    while (time.time() - initial_time < 2 * day) and (i < iterations):
        # Choose sphere
        spheres = arr.all_spheres
        sphere = spheres[random.randint(0, len(spheres) - 1)]
        cell = arr.cell_of_sphere(sphere)
        if algorithm == 'ECMC':
            # Choose direction
            direction = Direction.directions()[random.randint(0, 3)]  # x,y,+z,-z
            # perform step
            step = Step(sphere, xy_total_step if direction.dim != 2 else z_total_step, direction, arr.boundaries)
            i_cell, j_cell = cell.ind[:2]
            try:
                if record_displacements:
                    displacements.append(
                        displacements[-1] + arr.perform_total_step(i_cell, j_cell, step, record_displacements=True))
                    realizations.append(i + 1)
                else:
                    arr.perform_total_step(i_cell, j_cell, step)
            except Exception as err:
                if write:
                    files_interface.dump_spheres(arr.all_centers, str(i + 1) + '_err')
                raise err
        elif algorithm == 'MCMC':
            direction_angle = np.random.random() * 2 * np.pi
            arr.perform_MCMC_step(i_cell, j_cell, sphere,
                                  metropolis_step * np.array([np.cos(direction_angle), np.sin(direction_angle)]))
        if (i + 1) % (iterations / 100) == 0:
            print(str(100 * (i + 1) / iterations) + "%", end=", ", file=sys.stdout)
        if np.floor((time.time() - initial_time) / (hour / 2)) > realizations_saved:
            # save realization every half an hour
            realizations_saved += 1
            if write:
                files_interface.dump_spheres(arr.all_centers, str(i + 1))
        i += 1

    # save
    if record_displacements and write:
        np.savetxt(os.path.join(output_dir, 'Displacement'), np.array([realizations, displacements]).T)
    assert arr.legal_configuration()
    if write:
        files_interface.dump_spheres(arr.all_centers, str(i + 1))

    os.chdir(os.path.join(prefix, sim_name))
    if i >= iterations:
        if write:
            os.system('echo \'Finished ' + str(iterations) + ' iterations\' > FINAL_MESSAGE')
        sys.exit(0)
    else:
        os.system('echo \'\nElapsed time is ' + str(time.time() - initial_time) + '\' >> TIME_LOG')
        os.chdir(code_dir)
        sys.exit(7)  # any !=0 number


def main():
    sim_name = sys.argv[1]
    N, h, rhoH, ic, algorithm = params_from_name(sim_name)
    if ic == 'square':
        run_square(h, N, rhoH, algorithm)
    if ic == 'honeycomb':
        run_honeycomb(h, N, rhoH, algorithm)
    if ic == 'triangle':
        run_triangle(h, N, rhoH, algorithm)


if __name__ == "__main__":
    main()
