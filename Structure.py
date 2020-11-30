import copy
import numpy as np
import random
from enum import Enum
import warnings

epsilon = 1e-8


class Direction:
    def __init__(self, dim, sgn=1):
        self.dim = dim
        if dim == 2:
            self.sgn = sgn
        else:
            self.sgn = 1
            if sgn != 1:
                warnings.warn(
                    "Direction is not z but sgn was given, only positive xy steps are supported, wrong input. " + \
                    "Input correction to sgn=+1 was applied")
            if dim >= 3:
                raise ValueError("dim=0,1,2")

    @staticmethod
    def directions():
        return [Direction(0), Direction(1), Direction(2, 1), Direction(2, -1)]


class Sphere:

    def __init__(self, center, rad):
        """
        Create new sphere object
        :param center: d dimension of sphere center
        :param rad: scalar - radius of sphere
        """
        self.center = [c for c in center]  # list and not np array or tuple
        self.rad = rad

    @property
    def dim(self):
        """
        :return: d - the dimension of the Sphere instance
        """
        return len(self.center)

    def box_it(self, boundaries):
        """
        Put the sphere inside the boundaries of the simulation, usefull for cyclic boundary conditions
        :type boundaries: list
        """
        try:
            self.center = [x % e for x, e in zip(self.center, boundaries)]
        except RuntimeWarning:
            print(RuntimeWarning)

    def perform_step(self, direction: Direction, current_step, boundaries):
        dt = direction.sgn * current_step if direction.dim == 2 else current_step
        self.center[direction.dim] = self.center[direction.dim] + dt
        self.box_it(boundaries)


class BoundaryType(Enum):
    WALL = "RigidWall"
    CYCLIC = "CyclicBoundaryConditions"


class Metric:

    @staticmethod
    def dist_to_wall(sphere, total_step, direction: Direction, boundaries):
        """
        Figures out the distance for the next WALL boundary (ignoring CYCLIC boundaries)
        :type sphere: Sphere
        :param direction: direction of step
        :param total_step: magnitude of the step to be carried out
        :type boundaries: list
        :return: the minimal distance to the wall, and the wall.
        If there is no wall in a distance l, dist is inf and wall=[]
        """
        if direction.dim != 2:
            return float('inf')
        t = boundaries[2] - sphere.center[2] - sphere.rad - epsilon if direction.sgn == +1 else \
            sphere.center[2] - sphere.rad - epsilon
        return t if t < total_step else float('inf')

    @staticmethod
    def relevant_cyclic_transform_vecs(center1, boundaries, cut_off):
        l_x, l_y, c = boundaries[0], boundaries[1], center1
        vectors = [[0, 0]]  # 1
        x_down = c[0] - cut_off < 0
        y_down = c[1] - cut_off < 0
        x_up = c[0] + cut_off > l_x
        y_up = c[1] + cut_off > l_y
        if x_down:
            vectors.append([-l_x, 0])  # 2
            if y_down: vectors.append([-l_x, -l_y])  # 6
            if y_up: vectors.append([-l_x, l_y])  # 7
        if y_down: vectors.append([0, -l_y])  # 3
        if x_up:
            vectors.append([l_x, 0])  # 4
            if y_down: vectors.append([l_x, -l_y])  # 8
            if y_up: vectors.append([l_x, l_y])  # 9
        if y_up: vectors.append([0, l_y])  # 5
        return vectors

    @staticmethod
    def dist_to_collision(sphere1, other_spheres, total_step, direction: Direction, boundaries, cut_off=float('inf')):
        """
        Distance sphere1 would need to go in direction in order to collide with sphere2
        It is not implemented in the most efficient way,  because for cyclic xy we copy sphere2 8 times (for all
        cyclic boundaries) and then check for collision.
        It is implemented only for steps smaller then system size.
        :param sphere1: sphere about to move
        :type sphere1: Sphere
        :param sphere2: potential for collision
        :type sphere2: Sphere
        :param total_step: maximal step size. If dist for collision > total_step then dist_to_collision-->infty
        :param direction: in which sphere1 is to move
        :param boundaries: boundaries for the case some of them are cyclic boundary conditinos
        :type boundaries: list
        :return: distance for collision if the move is allowed, infty if move can not lead to collision
        """
        # for sphere2 in other_spheres:
        #     assert not Metric.overlap(sphere1, sphere2, boundaries), "Overlap between:\nSphere1: " + str(
        #         sphere1.center) + "\nSphere2: " + str(sphere2.center) + "\nBoundaries are: " + str(boundaries)
        c1 = sphere1.center
        vectors = Metric.relevant_cyclic_transform_vecs(c1, boundaries, cut_off)
        closest_sphere, closest_sphere_dist = [], float('inf')
        for sphere2 in other_spheres:
            c2 = sphere2.center
            sig_sq = (sphere1.rad + sphere2.rad) ** 2
            if direction.dim == 2:
                dz = (c2[2] - c1[2]) * direction.sgn
                if dz <= 0: continue
                for v in vectors:
                    discriminant = sig_sq - (c2[1] - c1[1] + v[1]) ** 2 - (c2[0] - c1[0] + v[0]) ** 2
                    if discriminant <= 0: continue
                    dist = dz - np.sqrt(discriminant)  # for non overlapping sphere dist>0 always, no need to check
                    if dist <= total_step and dist < closest_sphere_dist:
                        closest_sphere_dist = dist
                        closest_sphere = sphere2
            else:
                i, j = direction.dim, 1 - direction.dim
                sig_xy_sq = sig_sq - (c2[2] - c1[2]) ** 2
                if sig_xy_sq <= 0: continue
                for v in vectors:
                    # dx is in the direction of the step i, and dy in j direction
                    dx = c2[i] - c1[i] + v[i]
                    if dx <= 0: continue
                    discriminant = sig_xy_sq - (c2[j] - c1[j] + v[j]) ** 2
                    if discriminant <= 0: continue
                    dist = dx - np.sqrt(discriminant)
                    if dist <= total_step and dist < closest_sphere_dist:
                        closest_sphere_dist = dist
                        closest_sphere = sphere2
        return closest_sphere_dist, closest_sphere

    @staticmethod
    def cyclic_vec(boundaries, sphere1, sphere2):
        """
        :return: vector pointing from sphere2 to sphere1 ("sphere2-sphere1") shortest through cyclic boundaries
        """
        dx = np.array(sphere1.center) - sphere2.center  # direct vector
        vec = np.zeros(len(dx))
        vec[2] = dx[2]
        for i in range(2):
            l = boundaries[i]
            dxs = np.array([dx[i], dx[i] + l, dx[i] - l])
            vec[i] = dxs[np.argmin(dxs ** 2)]  # find shorter path through B.D.
        return vec

    @staticmethod
    def cyclic_dist(boundaries, sphere1, sphere2):
        dx = np.array(sphere1.center) - sphere2.center  # direct vector
        dsq = dx[2] ** 2
        for i in range(2):
            L = boundaries[i]
            dsq += min(dx[i] ** 2, (dx[i] + L) ** 2, (dx[i] - L) ** 2)  # find shorter path through B.D.
        return np.sqrt(dsq)

    @staticmethod
    def overlap(sphere1: Sphere, sphere2: Sphere, boundaries):
        """
        Test of two d-dimensional Sphere objects overlap
        :type sphere1: Sphere
        :type sphere2: Sphere
        :type boundaries: list
        :return: True if they overlap
        """
        if sphere1 == sphere2: return False
        delta = Metric.cyclic_dist(boundaries, sphere1, sphere2) - (sphere1.rad + sphere2.rad)
        if delta < -1e3 * epsilon:  # stabilizing calculation by forgiving some penetration
            return True
        else:
            if delta > 0:
                return False
            else:
                Warning("Spheres are epsilon close to each other, separating them")
                dr_hat = sphere1.center - np.array(sphere2.center)
                dr_hat = dr_hat / (np.linalg.norm(dr_hat))
                sphere1.center = sphere1.center + np.abs(delta) * dr_hat
                return False

    @staticmethod
    def spheres_overlap(spheres, boundaries):
        """
        :type spheres: list
        :param spheres: list of spheres to check direct_overlap between all couples
        :param boundaries: important for the cyclic boundary conditions
        :return: True if there are two overlapping spheres, even if they direct_overlap through boundary condition
        """
        for i in range(len(spheres)):
            for j in range(i):
                if Metric.overlap(spheres[i], spheres[j], boundaries):
                    return True
        return False

    @staticmethod
    def direct_overlap(spheres):
        for i in range(len(spheres)):
            for j in range(i):
                sphere1 = spheres[i]
                sphere2 = spheres[j]
                dist = np.linalg.norm(sphere1.center - sphere2.center)
                delta = dist - (sphere1.rad + sphere2.rad)
                if delta < -1e3 * epsilon:  # stabelize calculation by forgiving some penteration
                    return True
                else:
                    if delta > 0:
                        return False
                    else:
                        Warning("Spheres are epsilon close to each other, seperating them")
                        dr_hat = sphere1.center - np.array(sphere2.center)
                        dr_hat = dr_hat / (np.linalg.norm(dr_hat))
                        sphere1.center = sphere1.center + np.abs(delta) * dr_hat
                        return False


class Cell:

    def __init__(self, site, edges, ind=(), spheres=[]):
        """
        Cell object is a d-dimension cube in space, containing list of Sphere objects
        :param site: left bottom (in 3d back) corner of the cell
        :param edges: d dimension list of edges length for the cell
        :param ind: index for the cell inside a larger cell array
        :param spheres: list of d-dimension Sphere objects in the cell.
        """
        self.site = site
        self.edges = edges
        self.ind = ind
        self.spheres = spheres

    @property
    def dim(self):
        return len(self.site)

    def append(self, new_spheres):
        """
        Add spheres to the cell
        :param new_spheres: list of Sphere objects to be added to the cell
        """
        if type(new_spheres) == list:
            for sphere in new_spheres:
                # if sphere not in self.spheres:
                self.spheres.append(sphere)
        else:
            # assert type(new_spheres) == Sphere
            # if new_spheres not in self.spheres:
            self.spheres.append(new_spheres)

    def remove_sphere(self, spheres_to_remove):
        """
        Delete spheres from the cell
        :param spheres_to_remove: the Sphere objects to be removed (pointers!)
        """
        if type(spheres_to_remove) == list:
            for sphere in spheres_to_remove: self.spheres.remove(sphere)
        else:
            # assert type(spheres_to_remove) == Sphere
            self.spheres.remove(spheres_to_remove)

    def center_in_cell(self, sphere):
        """
        A Sphere object should be in a cell if its center is inside the physical d-dimension
        cube of the cell. Notice in the case dim(cell)!=dim(sphere), a sphere is considered
        inside a cell if its dim(cell) first components are inside the cell.
        :param sphere: the sphere to check whether should be considered in the new cell
        :type sphere: Sphere
        :return: True if the sphere should be considered inside the cell, false otherwise
        """
        for i in range(self.dim):
            x_sphere, x_site, edge = sphere.center[i], self.site[i], self.edges[i]
            # Notice this implementation instead of for x_... in zip() is for the case dim(sphere)!=dim(cell)
            if x_sphere <= x_site or x_sphere > x_site + edge:
                return False
        return True

    def random_generate_spheres(self, n_spheres, rads, l_z=3.0):
        """
        Generate n spheres inside the cell. If there are spheres in the cell already,
         it deletes the existing spheres. The algorithm is to randomaly
         generate their centers and check for direct_overlap. In order to save time,
         they are generated with a rad distance from the end of the cell, which
         might cause problems in the future
        :param n_spheres: number of spheres to be generated
        :param rads: rads of the different spheres
        :param extra_edges: if dim_spheres>dim(cell), there should be constraint on the extra
         dimension size. For 2d cells with 3d spheres confined between two walls, extra_edges=[h]
        """
        if type(rads) != list: rads = n_spheres * [rads]
        while True:  # do-while python implementation
            spheres = []
            for i in range(n_spheres):
                r = rads[i]
                center = np.concatenate((np.array(self.site) + [random.random() * e for e in self.edges],
                                         [random.random() * (l_z - 2 * r) + r]))
                spheres.append(Sphere(center, rads[i]))
            if not Metric.direct_overlap(spheres):
                break
        self.spheres = spheres

    def transform(self, new_site):
        """
        Transform the cell and all its spheres to the new_site.
        Might be usefull for boundary conditinos implementation
        :param new_site: new_site for the cell to transform to
        """
        dx = np.array(new_site) - self.site
        self.site = new_site
        for sphere in self.spheres:
            if len(dx) < sphere.dim:
                dx = [dx_ for dx_ in dx] + [0 for _ in range(sphere.dim - len(dx))]
                dx = np.array(dx)
            sphere.center = sphere.center + dx


class ArrayOfCells:

    def __init__(self, dim, boundaries, cells=[[]]):
        """
        :type boundaries: list
        :param dim: dimension of the array of cells. Doesn't have to be dimension of a single sphere.
        :param cells: list of cells defining the array, optional.
        """
        self.dim = dim
        self.cells = cells
        self.boundaries = boundaries
        self.n_rows = len(cells)
        self.n_columns = len(cells[0])
        self.all_spheres = []

    @property
    def all_centers(self):
        """
        :return: list of all the centers (d-dimension vectors) of all the Sphere objects in the simulation.
        """
        return [sphere.center for sphere in self.all_spheres]

    def overlap_2_cells(self, cell1: Cell, cell2: Cell):
        """
        Checks if the spheres in cell1 and cell2 are overlapping with each other. Does not check inside cell.
        :return: True if one of the spheres in cell1 direct_overlap with one of the spheres in cell2
        """
        if cell1 == [] or cell2 == []:
            return False
        spheres1 = cell1.spheres
        spheres2 = cell2.spheres
        for sphere1 in spheres1:
            for sphere2 in spheres2:
                if Metric.overlap(sphere1, sphere2, self.boundaries):
                    return True
        return False

    def overlap_2_cells_inds(self, i1: int, j1: int, i2: int, j2: int):
        """
        Checks if the spheres in cell1 and cell2 are overlapping with each other. Does not check inside cell.
        :return: True if one of the spheres in cell1 direct_overlap with one of the spheres in cell2
        """
        cell1 = self.cells[i1][j1]
        cell2 = self.cells[i2][j2]
        return self.overlap_2_cells(cell1, cell2)

    def cushioning_array_for_boundary_cond(self):
        """
        :return: array of cells of length n_row+2 n_column+2, for any CYCLIC boundary condition it is surrounded by the
        cells in the other end
        """
        if self.dim != 2: raise Exception("dim!=2 not implemented yet")
        n_rows = self.n_rows
        n_columns = self.n_columns
        cells = [[Cell((), []) for _ in range(n_columns + 2)] for _ in range(n_rows + 2)]
        for i in range(n_rows):  # 1 < (i + 1, j+1) < n
            for j in range(n_columns):
                cells[i + 1][j + 1] = self.cells[i][j]
        l_x, l_y = self.boundaries[0:2]
        for i in range(n_rows):  # 1 < i + 1 < n
            c0 = copy.deepcopy(self.cells[i][n_columns - 1])
            c0.transform(c0.site + np.array([-l_x, 0]))
            c1 = copy.deepcopy(self.cells[i][0])
            c1.transform(c1.site + np.array([l_x, 0]))
            cells[i + 1][0] = c0
            cells[i + 1][n_columns + 1] = c1
        for j in range(n_columns):  # 1 < j + 1 < n
            c0 = copy.deepcopy(self.cells[n_rows - 1][j])
            c0.transform(c0.site + np.array([0, -l_y]))
            c1 = copy.deepcopy(self.cells[0][j])
            c1.transform(c1.site + np.array([0, l_y]))
            cells[0][j + 1] = c0
            cells[n_rows + 1][j + 1] = c1
        c = copy.deepcopy(self.cells[n_rows - 1][n_columns - 1])
        c.transform(c.site + np.array([-l_x, -l_y]))
        cells[0][0] = c
        c = copy.deepcopy(self.cells[0][0])
        c.transform(c.site + np.array([l_x, l_y]))
        cells[n_rows + 1][n_columns + 1] = c
        c = copy.deepcopy(self.cells[0][n_columns - 1])
        c.transform(c.site + np.array([-l_x, l_y]))
        cells[n_rows + 1][0] = c
        c = copy.deepcopy(self.cells[n_rows - 1][0])
        c.transform(c.site + np.array([l_x, -l_y]))
        cells[0][n_columns + 1] = c
        return ArrayOfCells(self.dim, self.boundaries, cells)

    @staticmethod
    def cyclic_indices(i, j, n_rows, n_columns):
        ip1 = int((i + 1) % n_rows)
        jp1 = int((j + 1) % n_columns)
        im1 = int((i - 1) % n_rows)
        jm1 = int((j - 1) % n_columns)
        return ip1, jp1, im1, jm1

    def neighbors(self, i, j):
        ip1, jp1, im1, jm1 = ArrayOfCells.cyclic_indices(i, j, self.n_rows, self.n_columns)
        neighbor_cells = [self.cells[ip1][jm1], self.cells[ip1][j],
                          self.cells[ip1][jp1], self.cells[i][jp1],
                          self.cells[i][jm1], self.cells[im1][jm1],
                          self.cells[im1][j], self.cells[im1][jp1]]
        # First four are top and to the right, then to the left and bottom ones.
        # For efficient legal_configuration implementation
        return neighbor_cells

    def legal_configuration(self):
        """
        :return: True if there are no overlapping spheres in the configuration
        """
        if self.cells == []: return True
        if self.dim != 2:
            raise (Exception('Only d=2 supported!'))
        n_rows, n_columns = self.n_rows, self.n_columns
        for i in range(n_rows):
            for j in range(n_columns):
                cell = self.cells[i][j]
                if Metric.spheres_overlap(cell.spheres, self.boundaries):
                    return False
                for sphere in cell.spheres:
                    assert cell.center_in_cell(sphere), "sphere is missing from cell"
                for sphere in cell.spheres:
                    c_z = sphere.center[2]
                    r = sphere.rad
                    if c_z - r < -epsilon or c_z + r > self.boundaries[2] + epsilon:
                        return False
                for neighbor in self.neighbors(i, j):
                    if self.overlap_2_cells(cell, neighbor):
                        return False
        return True

    def cell_from_ind(self, ind):
        """
        :param ind: tuple (i,j,...)
        :return: cell at ind
        """
        cell = self.cells
        for i in ind: cell = cell[i]
        return cell

    def random_generate_spheres(self, n_spheres_per_cell, rad, l_z=3.0):
        if type(rad) != list: rad = n_spheres_per_cell * [rad]
        while True:
            for i in range(len(self.cells)):
                for j in range(len(self.cells[i])):
                    cell = self.cells[i][j]
                    cell.random_generate_spheres(n_spheres_per_cell, rad, l_z)
            if self.legal_configuration():
                return

    def generate_spheres_in_cubic_structure(self, n_spheres_per_cell, rad):
        if type(rad) != list: rad = n_spheres_per_cell * [rad]
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                cell = self.cells[i][j]
                dx, dy, dz = epsilon, epsilon, epsilon
                x0, y0 = cell.site
                max_r = 0
                for k in range(n_spheres_per_cell):
                    r = rad[k]
                    if r > max_r: max_r = r
                    center = (x0 + dx + r, y0 + dy + r, dz + r)
                    cell.append(Sphere(center, r))
                    dx += 2 * r + epsilon
                    if (k < n_spheres_per_cell - 1) and (dx + 2 * rad[k + 1] > cell.edges[0]):
                        dx = 0
                        dy += 2 * max_r + epsilon
                        max_r = 0
                    if (k < n_spheres_per_cell - 1) and (dy + 2 * rad[k + 1] > cell.edges[1]):
                        break
        assert self.legal_configuration()

    def append_sphere(self, spheres):
        if type(spheres) != list:
            # assert type(spheres) == Sphere
            spheres = [spheres]
        cells = []
        for sphere in spheres:
            self.all_spheres.append(sphere)
            sp_added_to_cell = False
            for i in range(len(self.cells)):
                for j in range(len(self.cells[i])):
                    c = self.cells[i][j]
                    if c.center_in_cell(sphere):
                        c.append(sphere)
                        cells.append(c)
                        sp_added_to_cell = True
                        break
            if not sp_added_to_cell:
                raise ValueError("A sphere was not added to any of the cells")
        if len(cells) == 1:
            return cells[0]
        return cells

    @staticmethod
    def spheres_in_triangular(n_row, n_col, rad, l_x, l_y):
        assert type(rad) != list, "list of different rads is not supported for initial condition triangular"
        ax = l_x / (n_col + 1 / 2)
        ay = l_y / (n_row + 1)
        assert ax >= 2 * rad and ay / np.cos(
            np.pi / 6) >= 2 * rad, "ferro triangle initial conditions are not defined for a<2*r, too many spheres"
        spheres = []
        for i in range(n_row):
            for j in range(n_col):
                if i % 2 == 0:
                    xj = (1 + epsilon) * rad + ax * j
                else:
                    xj = (1 + epsilon) * rad + ax * (j + 1 / 2)  # cos(pi / 3) = 1 / 2
                yi = (1 + epsilon) * rad + ay * i
                spheres.append(Sphere([xj, yi, rad], rad))
        return spheres

    def translate(self, vec):
        """
        Move all spheres
        :param vec: vector of translation.
        :return: transfered spheres, after removing all spheres from the array of cells
        """
        transferred_spheres = []
        vec = np.array(vec)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                c = self.cells[i][j]
                for s in c.spheres:
                    for k in range(min(len(s.center), len(vec))):
                        s.center[k] += vec[k]
                    transferred_spheres.append(s)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                c = self.cells[i][j]
                c.spheres = []
        # self.append_sphere(transferred_spheres)
        return transferred_spheres
