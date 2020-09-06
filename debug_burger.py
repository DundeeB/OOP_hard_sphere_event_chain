from post_process import *

sim_path = '../post_process/from_ATLAS2.0/N=900_h=0.8_rhoH=0.81_AF_triangle_ECMC'
# sim_path = '../post_process/from_ATLAS2.0/N=8100_h=0.8_rhoH=0.86_AF_square_ECMC'
load = WriteOrLoad(output_dir=sim_path)
l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = load.load_Input()
# load.boundaries = CubeBoundaries([l_x, l_y], 2 * [BoundaryType.CYCLIC])
# sp, spheres_i = load.last_spheres()
# event_2d_cells = Event2DCells(edge=edge, n_rows=n_row, n_columns=n_col)
# event_2d_cells.append_sphere([Sphere(x, rad=1.0) for x in sp])

psi14 = PsiMN(sim_path, 1, 4)
psi14.calc_order_parameter()
psi14.write(write_correlation=False, write_vec=True)

N = rho_H * l_x * l_y * l_z / np.power(2 * rad, 3)
A = l_x * l_y
a = np.sqrt(A / N)
a1 = np.array([a, 0])
a2 = np.array([0, a])
burger = BurgerField(sim_path, a1, a2, psi14)
burger.calc_order_parameter()
burger.write()
