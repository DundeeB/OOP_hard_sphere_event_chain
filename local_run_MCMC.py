import os
import sys
from EventChainActions import *
from SnapShot import WriteOrLoad
import re
import time
from send_parametric_runs import *
import matplotlib.pyplot as plt


def plt_arr(arr, label='initial conditions', plot_boundaries=False):
    x = [c[0] for c in arr.all_centers]
    y = [c[1] for c in arr.all_centers]
    if plot_boundaries:
        b = arr.boundaries
        plt.plot([0, 0, b[0], b[0], 0], [0, b[1], b[1], 0, 0], label='cylic boundaries')
    plt.plot(x, y, '.', label=label)
    plt.axis('equal')


epsilon = 1e-8
prefix = '/storage/ph_daniel/danielab/ECMC_simulation_results3.0/'

N, h, rhoH, ic, algorithm = 30**2, 0.8, 0.8, 'square', 'LMC'

# run_honeycomb
if ic == 'honeycomb':
    n_row = int(np.sqrt(N))
    n_col = n_row
    r = 1
    sig = 2 * r
    # build input parameters for cells
    a_dest = sig * np.sqrt(2 / (rhoH * (1 + h) * np.sin(np.pi / 3)))
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
    initial_arr.scale_xy(np.sqrt(rho_H_new / rhoH))
    assert initial_arr.edge > sig
# run_square
elif ic == 'square':
    n_row = int(np.sqrt(N))
    n_col = n_row  # Square initial condition for n_row!=n_col is not implemented...
    r, sig = 1.0, 2.0
    A = N * sig ** 2 / (rhoH * (1 + h))
    a = np.sqrt(A / N)
    n_row_cells, n_col_cells = int(np.sqrt(A) / (a * np.sqrt(2))), int(np.sqrt(A) / (a * np.sqrt(2)))
    e = np.sqrt(A / (n_row_cells * n_col_cells))
    assert e > sig, "Edge of cell is: " + str(e) + ", which is smaller than sigma."
    initial_arr = Event2DCells(edge=e, n_rows=n_row_cells, n_columns=n_col_cells, l_z=(h + 1) * sig)
    initial_arr.generate_spheres_in_AF_square(n_row, n_col, r)

# run_sim
iterations = int(N * 1e4)
rad = 1
a_free = (1 / rhoH - np.pi / 6) ** (1 / 3) * 2 * rad  # ((V-N*4/3*pi*r^3)/N)^(1/3)
if algorithm == 'ECMC':
    xy_total_step = a_free * np.sqrt(N)
    z_total_step = h * (2 * rad) * np.pi / 15  # irrational for the spheres to cover most of the z options
elif algorithm == 'LMC':
    metropolis_step = a_free * (np.pi / 3) / 8
arr = initial_arr
print("\n\nSimulation: N=" + str(N) + ", rhoH=" + str(rhoH) + ", h=" + str(h), file=sys.stdout)
print("Lx=" + str(initial_arr.l_x) + ", Ly=" + str(initial_arr.l_y), file=sys.stdout)

# Run loops
day = 86400  # seconds
hour = 3600  # seconds
plt_arr(arr, plot_boundaries=True)
counter = 0
for i in range(10 ** 3):
    # Choose sphere
    spheres = arr.all_spheres
    i_sp = random.randint(0, len(spheres) - 1)
    sphere = spheres[i_sp]
    cell = arr.cell_of_sphere(sphere)
    i_cell, j_cell = cell.ind[:2]
    if algorithm == 'ECMC':
        # Choose direction
        direction = Direction.directions()[random.randint(0, 3)]  # x,y,+z,-z
        # perform step
        step = Step(sphere, xy_total_step if direction.dim != 2 else z_total_step, direction, arr.boundaries)
        try:
            arr.perform_total_step(i_cell, j_cell, step)
        except Exception as err:
            raise err
    elif algorithm == 'LMC':
        theta, phi = np.random.random() * np.pi, np.random.random() * 2 * np.pi
        try:
            old_arr = copy.deepcopy(arr)
            old_sphere = old_arr.all_spheres[i_sp]
            accepted_move = arr.perform_LMC_step(i_cell, j_cell, sphere, metropolis_step * np.array(
                [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]))
        except IndexError as err:
            plt_arr(arr, label='realization ' + str(i) + ' with error')
            raise
        if accepted_move:
            counter += 1
    i += 1
assert arr.legal_configuration()
print("Acceptance ratio = " + str(counter / i))
plt_arr(arr, label='realization ' + str(i))
plt.legend()
plt.show()
