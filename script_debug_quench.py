from EventChainActions import *
from SnapShot import WriteOrLoad
import numpy as np

# r, sig, l_x, l_y, l_z = 1.0, 2.0, 132.63770271305015, 159.65649400644926, 2.0 * (1 + 0.8)
r, sig, l_x, l_y, l_z = 1.0, 2.0, 150, 170, 2.0 * (1 + 0.8)
edge = 3*r
n_rows, n_columns = int(np.ceil(l_y / edge)), int(np.ceil(l_x / edge))
l_x, l_y = n_columns * edge, n_rows * edge

initial_arr = Event2DCells(edge, n_rows, n_columns)
initial_arr.boundaries = CubeBoundaries([l_x, l_y], 2 * [BoundaryType.CYCLIC])
initial_arr.l_x = l_x
initial_arr.l_y = l_y
initial_arr.add_third_dimension_for_sphere(l_z)
file_interface = WriteOrLoad('.', initial_arr.boundaries)
initial_arr.append_sphere([Sphere(c, r) for c in file_interface.last_spheres()[0]])
assert initial_arr.legal_configuration()
# desired_rho = 0.89
# initial_arr.quench(desired_rho)
desired_h = 0.75
initial_arr.z_quench(desired_h)
