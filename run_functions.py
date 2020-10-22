#!/Local/cmp/anaconda3/bin/python -u
import os
import sys
from EventChainActions import *
from SnapShot import WriteOrLoad
import re
import time
from send_parametric_runs import *

epsilon = 1e-8
# prefix = '/storage/ph_daniel/danielab/ECMC_simulation_results2.0/'
prefix = 'C:\\Users\\Daniel Abutbul\\OneDrive - Technion\\simulation-results'


def run_honeycomb(h, n_row, n_col, rho_H, **kwargs):
    # More physical properties calculated from Input
    N = n_row * n_col
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_triangle_ECMC'
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, h, rho_H, sim_name, **kwargs)
        # when continuing from restart there shouldn't be use of initial arr
    else:
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


def run_square(h, n_row, n_col, rho_H, **kwargs):
    # More physical properties calculated from Input
    N = n_row * n_col
    sim_name = 'N=' + str(N) + '_h=' + str(h) + '_rhoH=' + str(rho_H) + '_AF_square_ECMC'
    output_dir = prefix + sim_name
    if os.path.exists(output_dir):
        return run_sim(np.nan, N, h, rho_H, sim_name, **kwargs)
        # when continuing from restart there shouldn't be use of initial arr
    else:
        r, sig = 1, 2
        A = N * sig ** 2 / (rho_H * (1 + h))
        assert n_row == n_col, "Square initial condition for n_row!=n_col is no implemented..."
        N = n_col * n_row
        a = np.sqrt(A / N)
        n_row_cells, n_col_cells = int(np.sqrt(A) / (a * np.sqrt(2))), int(np.sqrt(A) / (a * np.sqrt(2)))
        e = np.sqrt(A / (n_row_cells * n_col_cells))
        initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells, l_z=(h + 1) * sig)
        initial_arr.generate_spheres_in_AF_square(n_row, n_col, r)
        assert initial_arr.edge > sig, "Edge of cell is: " + str(initial_arr.edge) + ", which is smaller than sigma."

        return run_sim(initial_arr, N, h, rho_H, sim_name, **kwargs)


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

    initial_arr = Event2DCells(edge=edge, n_rows=n_row, n_columns=n_col, l_z=l_z)
    initial_arr.boundaries = [l_x, l_y, l_z]
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


def run_sim(initial_arr, N, h, rho_H, sim_name, iterations=None, record_displacements=False, write=True):
    if iterations is None:
        iterations = int(N * 1e4)
    rad = 1
    a_free = (1 / rho_H - np.pi / 6) ** (1 / 3) * 2 * rad  # ((V-N*4/3*pi*r^3)/N)^(1/3)
    xy_total_step = a_free * np.sqrt(N)
    z_total_step = h * rad * np.pi / 3  # make it around h*rad in order for the spheres to cover most of the z otions

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
    initial_time = time.time()
    day = 86400  # seconds
    i = last_ind
    if record_displacements:
        displacements = [0]
        realizations = [i]
    while time.time() - initial_time < day and i < iterations:
        # Choose sphere
        while True:
            i_all_cells = random.randint(0, len(arr.all_cells) - 1)
            cell = arr.all_cells[i_all_cells]
            if len(cell.spheres) > 0:
                break
        sphere = cell.spheres[random.randint(0, len(cell.spheres) - 1)]

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
        except:
            files_interface.dump_spheres(arr.all_centers, str(i + 1) + '_err')
            raise
        if (i + 1) % (iterations / 100) == 0:
            print(str(100 * (i + 1) / iterations) + "%", end=", ", file=sys.stdout)
        i += 1

    # save
    if record_displacements and write:
        np.savetxt(os.path.join(output_dir, 'Displacement'), np.array([realizations, displacements]).T)
    assert arr.legal_configuration()
    if write:
        files_interface.dump_spheres(arr.all_centers, str(i + 1))

    if i >= iterations:
        if write:
            os.system('echo \'Finished ' + str(iterations) + ' iterations\' > FINAL_MESSAGE')
    else:  # resend the simulation
        os.system('echo \'\nElapsed time is ' + str(time.time() - initial_time) + '\' >> TIME_LOG')
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
    return 0 if not record_displacements else realizations, displacements


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
