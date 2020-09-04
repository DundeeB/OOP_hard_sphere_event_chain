from post_process import *

sim_path = '../post_process/from_ATLAS2.0/N=900_h=0.8_rhoH=0.81_AF_triangle_ECMC'
spheres_i = '836390'
load = WriteOrLoad(output_dir=sim_path)
l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = load.load_Input()
load.boundaries = CubeBoundaries([l_x, l_y], 2 * [BoundaryType.CYCLIC])
