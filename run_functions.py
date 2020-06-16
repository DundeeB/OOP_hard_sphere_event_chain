#!/Local/cmp/anaconda3/bin/python -u
import os
import sys
from EventChainActions import *
from SnapShot import WriteOrLoad
import re
import time
from send_parametric_runs import *

epsilon = 1e-8
prefix = '/storage/ph_daniel/danielab/ECMC_simulation_results2.0/'


def run_honeycomb(h, n_row, n_col, rho_H):
    # More physical properties calculated from Input
    N = n_row * n_col
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_triangle_ECMC'
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, h, rho_H, sim_name)
        # when continuing from restart there shouldn't be use of initial arr

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

    initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells)
    initial_arr.add_third_dimension_for_sphere((h + 1) * sig)
    initial_arr.generate_spheres_in_AF_triangular_structure(n_row, n_col, r)
    initial_arr.scale_xy(np.sqrt(rho_H_new / rho_H))
    assert initial_arr.edge > sig

    return run_sim(initial_arr, N, h, rho_H, sim_name)


def run_square(h, n_row, n_col, rho_H):
    # More physical properties calculated from Input
    N = n_row * n_col
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_square_ECMC'
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, h, rho_H, sim_name)
        # when continuing from restart there shouldn't be use of initial arr
    else:
        r, sig = 1, 2
        A = N * sig ** 2 / (rho_H * (1 + h))
        e = np.sqrt(A / (n_col * n_row))
        if e <= sig:
            e *= np.sqrt(2)
            if n_row % 2 == 0:
                n_row = int(n_row / 2)
            else:
                if n_col % 2 == 0:
                    n_col = int(n_col / 2)
                else:
                    raise Exception("Not implemented square initial condition with odd rows and columns")
        initial_arr = Event2DCells(edge=e, n_rows=n_row, n_columns=n_col)
        initial_arr.add_third_dimension_for_sphere((h + 1) * sig)
        initial_arr.generate_spheres_in_AF_square(n_row, n_col, r)
        assert initial_arr.edge > sig, "Edge of cell is: " + str(initial_arr.edge) + ", which is smaller than sigma."

        return run_sim(initial_arr, N, h, rho_H, sim_name)


def run_z_quench(origin_sim, desired_h):
    physical_info = re.split('[=_]', origin_sim)
    N, h, rho_H = int(physical_info[1]), float(physical_info[3]), float(physical_info[5])
    desired_rho = rho_H * (h + 1) / (desired_h + 1)
    sim_name = 'N=' + str(N) + '_h=' + str(desired_h) + '_rhoH=' + str(desired_rho) + '_from_zquench_ECMC'
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, desired_h, desired_rho, sim_name)
        # when continuing from restart there shouldn't be use of initial arr

    origin_sim_path = prefix + origin_sim
    orig_sim_files_interface = WriteOrLoad(origin_sim_path, boundaries=[])
    l_x, l_y, l_z, rad, _, edge, n_row, n_col = orig_sim_files_interface.load_Input()

    if (edge * n_row != l_y or edge * n_col != l_x) and (
            np.abs(edge * n_row - l_y) < epsilon and np.abs(edge * n_col - l_x) < epsilon):
        l_x, l_y = edge * n_col, edge * n_row
    assert n_col * edge == l_x and n_row * edge == l_y, \
        "Did not recover consistence system size and cells size.\n Chosen parameters are:\n" \
        + "edge=" + str(edge) + "\nn_row=" + str(n_row) + "\nn_col=" + str(
            n_col) + "\nWhile system size is:\nl_x=" + str(l_x) + "\nl_y=" + str(l_y)

    initial_arr = Event2DCells(edge=edge, n_rows=n_row, n_columns=n_col)
    initial_arr.boundaries = CubeBoundaries([l_x, l_y], 2 * [BoundaryType.CYCLIC])
    initial_arr.add_third_dimension_for_sphere(l_z)
    centers, ind = orig_sim_files_interface.last_spheres()
    assert initial_arr.edge > 2 * rad
    initial_arr.append_sphere([Sphere(c, rad) for c in centers])
    assert initial_arr.legal_configuration()
    assert len(initial_arr.all_spheres) == N, "Some spheres are missing. number of spheres added: " + str(
        len(initial_arr.all_spheres))
    try:
        initial_arr.z_quench((desired_h + 1) * (2 * rad))
    except:
        orig_sim_files_interface.boundaries = initial_arr.boundaries
        orig_sim_files_interface.dump_spheres(initial_arr.all_centers, 'z_quench_failed_lz=' + str(initial_arr.l_z))
        raise
    print('Taken from' + origin_sim + ', file ' + str(ind) + '. zQuenched successfully to rho=' +
          str(desired_rho) + ', h=' + str(desired_h))
    return run_sim(initial_arr, N, desired_h, desired_rho, sim_name)


def run_sim(initial_arr, N, h, rho_H, sim_name):
    # N_iteration = int(N * 1e4)
    iterations = int(N)
    rad = 1
    a_free = (1 / rho_H - np.pi / 6) * 2 * rad  # (V-N*4/3*pi*r^3)/N
    total_step = a_free * np.sqrt(N)

    # Initialize View and folder, add spheres
    code_dir = os.getcwd()
    output_dir = prefix + sim_name
    batch = output_dir + '/batch'
    if os.path.exists(output_dir):
        files_interface = WriteOrLoad(output_dir, np.nan)
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = files_interface.load_Input()
        boundaries = CubeBoundaries([l_x, l_y, l_z], 2 * [BoundaryType.CYCLIC] + [BoundaryType.WALL])
        files_interface.boundaries = boundaries
        last_centers, last_ind = files_interface.last_spheres()
        sp = [Sphere(tuple(c), rad) for c in last_centers]
        # construct array of cells and fill with spheres
        arr = Event2DCells(edge=edge, n_rows=n_row, n_columns=n_col)
        arr.add_third_dimension_for_sphere(l_z)
        arr.append_sphere(sp)
        sys.stdout = open(batch, "a")
        print("\n-----------\nSimulation with same parameters exist already, continuing from last file.\n",
              file=sys.stdout)
    else:
        files_interface = WriteOrLoad(output_dir, initial_arr.boundaries)
        os.mkdir(output_dir)
        sys.stdout = open(batch, "a")
        arr = initial_arr
        files_interface.dump_spheres(arr.all_centers, 'Initial Conditions')
        files_interface.save_Input(arr.all_spheres[0].rad, rho_H, arr.edge, arr.n_rows, arr.n_columns)
        last_ind = 0  # count starts from 1 so 0 means non exist yet and the first one will be i+1=1
        # Simulation description
        print("\n\nSimulation: N=" + str(N) + ", rhoH=" + str(rho_H) + ", h=" + str(h), file=sys.stdout)
        print("N_iterations=" + str(iterations) +
              ", Lx=" + str(initial_arr.l_x) + ", Ly=" + str(initial_arr.l_y), file=sys.stdout)

    os.chdir(output_dir)

    # Run loops
    initial_time = time.time()
    day = 60 * 60 * 24  # sec=1
    i = last_ind
    # while time.time() - initial_time < day and i < N_iteration:
    while time.time() - initial_time < 60 and i < iterations:
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
        phi = np.random.random() * 2 * np.pi
        v_hat = (np.cos(phi) * np.sin(t), np.sin(phi) * np.sin(t), np.cos(t))
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)

        # perform step
        step = Step(sphere, total_step, v_hat, arr.boundaries)
        i_cell, j_cell = cell.ind[:2]
        arr.perform_total_step(i_cell, j_cell, step)
        if (i + 1) % (iterations / 100) == 0:
            print(str(100 * (i + 1) / iterations) + "%", end=", ", file=sys.stdout)
        i += 1

    # save
    assert arr.legal_configuration()
    files_interface.dump_spheres(arr.all_centers, str(i + 1))

    if i >= iterations:
        os.system('echo \'Finished ' + str(iterations) + ' iterations\' > FINAL_MESSAGE')
    else:  # resend the simulation
        os.system('echo \'\nElapsed time is ' + str(time.time() - initial_time) + '\' > TIME_LOG')
        os.chdir(code_dir)
        resend_flag = False
        n_factor = int(np.sqrt(N / 900))
        ic = re.split('_', sim_name)[4]
        if ic == 'square':
            return send_single_run_envelope(h, 30 * n_factor, 30 * n_factor, rho_H, 'square')
        if ic == 'triangle':
            return send_single_run_envelope(h, 50 * n_factor, 18 * n_factor, rho_H, 'honeycomb')
        if ic == 'zquench':
            return quench_single_run_envelope('zquench', sim_name,
                                              desired_rho_or_h=h)  # notice run sim is sent after z-quench has succeeded
        assert resend_flag, "Simulation did not resend. Initial conditions: " + ic
    return 0


def main():
    args = sys.argv[1:]
    if len(args) == 5:
        h, n_row, n_col, rho_H = [float(x) for x in args[0:4]]
        n_col, n_row = int(n_col), int(n_row)
        if args[-1] == 'square':
            run_square(h, n_row, n_col, rho_H)
        else:
            if args[-1] == 'honeycomb':
                run_honeycomb(h, n_row, n_col, rho_H)
    else:
        action, other_sim_dir, desired_h = args[0], args[1], float(args[2])
        if action == 'zquench':
            run_z_quench(other_sim_dir, desired_h)


if __name__ == "__main__":
    main()
