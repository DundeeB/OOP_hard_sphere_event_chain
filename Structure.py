import copy
import numpy as np
import random
from enum import Enum

epsilon = 1e-8


class Direction:
    def __init__(self, dim, sgn=1):
        self.dim, self.sgn = dim, sgn

    def flip(self):
        self.sgn = -1 * self.sgn


class Sphere:

    def __init__(self, center, rad):
        """
        Create new sphere object
        :param center: d dimension of sphere center
        :param rad: scalar - radius of sphere
        """
        self.center = np.array(center)
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
        :type boundaries: CubeBoundaries
        """
        try:
            self.center = [x % e for x, e in zip(self.center, boundaries.edges)]
        except RuntimeWarning:
            print(RuntimeWarning)

    def perform_step(self, direction: Direction, current_step, boundaries):
        dt = direction.sgn * current_step if direction.dim == 2 else current_step
        self.center[direction.dim] = self.center[direction.dim] + dt

        self.box_it(boundaries)


class BoundaryType(Enum):
    WALL = "RigidWall"
    CYCLIC = "CyclicBoundaryConditions"


class CubeBoundaries:

    def __init__(self, edges, boundaries_type):
        """        CubeBoundaries constructor create new boundaries for the simulation.
        :param edges: list of edge per dimension for the simulation
        :param boundaries_type: list of boundary conditions, where the i'th arg fits the planes perpendicular to the
        i'th unit vector. For example, boundary_type[0] = BoundaryType.CYCLIC means the plane perpendicular to x-hat,
        in 1D the two ends of the rope, in 2D the planes y=0,1 and in 3D the planes yz (x=0,1), are cyclic.
        """
        assert len(edges) == len(boundaries_type)
        for bound in boundaries_type:
            assert type(bound) == BoundaryType
        self.edges = edges
        self.boundaries_type = boundaries_type
        self.dim = len(edges)

    @property
    def vertices(self):
        """
        :return: list of sites/vertices of the cube
        """
        e0 = self.edges[0]
        if self.dim == 1:
            return [(0,), (e0,)]
        e1 = self.edges[1]
        if self.dim == 2:
            return [(0, 0), (e0, 0), (0, e1), (e0, e1)]
        else:
            e2 = self.edges[2]  # dim==3
            return [(0, 0, 0), (e0, 0, 0), (0, e1, 0), (0, 0, e2),
                    (0, e1, e2), (e0, 0, e2), (e0, e1, 0), (e0, e1, e2)]

    @property
    def planes(self):
        vs = self.vertices
        if self.dim == 1:
            return [(vs[0],), (vs[1],)]
        if self.dim == 2:
            return [(vs[0], vs[1]), (vs[0], vs[2]),
                    (vs[1], vs[3]), (vs[2], vs[3])]
        else:  # dim==3
            return [(vs[0], vs[2], vs[1], vs[6]),  # xy
                    (vs[0], vs[2], vs[3], vs[4]),  # yz
                    (vs[0], vs[3], vs[1], vs[5]),  # xz
                    (vs[4], vs[7], vs[3], vs[5]),  # xy (z=1)
                    (vs[1], vs[5], vs[6], vs[7]),  # yz (x=1)
                    (vs[2], vs[4], vs[6], vs[7])]  # xz (y=1)

    @property
    def planes_type(self):
        """
        :return: list of boundary conditions, the i'th boundary condition is for i'th wall in self.walls
        """
        bc0 = self.boundaries_type[0]
        if self.dim == 1: return 2 * [bc0]
        bc1 = self.boundaries_type[1]
        if self.dim == 2:
            return [bc1, bc0, bc0, bc1]
        else:  # dim=3
            bc2 = self.boundaries_type[2]
            # xy yz xz xy yz xz
            return 2 * [bc2, bc0, bc1]

    def boundary_transformed_vectors(self):
        l_x = self.edges[0]
        l_y = self.edges[1]
        if self.dim == 3 and self.boundaries_type[2] == BoundaryType.CYCLIC: raise Exception(
            'z wall CYCLIC not supported')
        x_cyclic = (self.boundaries_type[0] == BoundaryType.CYCLIC)
        y_cyclic = (self.boundaries_type[1] == BoundaryType.CYCLIC)
        both_cyclic = (x_cyclic and y_cyclic)
        vectors = [(0, 0)]
        if x_cyclic:
            for vec in [(l_x, 0), (-l_x, 0)]:
                vectors.append(vec)
        if y_cyclic:
            for vec in [(0, l_y), (0, -l_y)]:
                vectors.append(vec)
        if both_cyclic:
            for vec in [(l_x, l_y), (l_x, -l_y), (-l_x, -l_y), (-l_x, l_y)]:
                vectors.append(vec)
        vectors = [np.array(v) for v in vectors]
        if self.dim == 3: vectors = np.array([[x for x in v] + [0] for v in vectors])
        return vectors


class Metric:

    @staticmethod
    def dist_to_wall(sphere, total_step, direction: Direction, boundaries):
        """
        Figures out the distance for the next WALL boundary (ignoring CYCLIC boundaries)
        :type sphere: Sphere
        :param direction: direction of step
        :param total_step: magnitude of the step to be carried out
        :type boundaries: CubeBoundaries
        :return: the minimal distance to the wall, and the wall.
        If there is no wall in a distance l, dist is inf and wall=[]
        """
        pos = np.array(sphere.center)
        r = sphere.rad
        if direction.dim != 2:
            return float('inf')
        l = boundaries.edges[2] - pos[2] - r - epsilon if direction.sgn == +1 else pos[2] - r - epsilon
        return l if l < total_step else float('inf')

    @staticmethod
    def dist_to_collision(sphere1, sphere2, total_step, direction: Direction, boundaries):
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
        :type boundaries: CubeBoundaries
        :return: distance for collision if the move is allowed, infty if move can not lead to collision
        """
        assert not Metric.overlap(sphere1, sphere2, boundaries), "Overlap between:\nSphere1: " + str(
            sphere1.center) + "\nSphere2: " + str(sphere2.center) + "\nBoundaries are: " + str(boundaries.edges)
        c1, c2 = sphere1.center, sphere2.center
        sig_sq = (sphere1.rad + sphere2.rad) ** 2
        if direction.dim == 2:
            dz = (c2[2] - c1[2]) * direction.sgn
            if dz < 0:
                return float('inf')
            effective_sig_sq = sig_sq - (c2[1] - c1[1]) ** 2 - (c2[0] - c1[0]) ** 2
            if effective_sig_sq < 0:
                return float('inf')
            t = dz - np.sqrt(effective_sig_sq)
            if t < 0 or t > total_step:
                return float('inf')
            return t
        x1, x2, y1, y2, z1, z2, lx, ly = c1[direction.dim], c2[direction.dim], c1[1 - direction.dim], c2[
            1 - direction.dim], c1[2], c2[2], boundaries.edges[direction.dim], boundaries.edges[1 - direction.dim]
        # now assume the step is in the x direction because I reorganize x and y
        effective_sigs_sq = [sig_sq - (y2 - y1 - l) ** 2 - (z2 - z1) ** 2 for l in [0, ly, -ly]]
        possible_ts = [x2 - x1 - l - np.sqrt(effective_sig_sq) for l in [0, lx, -lx] for effective_sig_sq in
                       effective_sigs_sq if effective_sig_sq > 0]
        ts = [t for t in possible_ts if t > 0 and t < total_step]
        if len(ts) == 0:
            return float('inf')
        return min(ts)

    @staticmethod
    def cyclic_vec(boundaries, sphere1, sphere2):
        """
        :return: vector pointing from sphere2 to sphere1 ("sphere2-sphere1") shortest through cyclic boundaries
        """
        dx = np.array(sphere1.center) - sphere2.center  # direct vector
        vec = np.zeros(len(dx))
        for i, b in enumerate(boundaries.boundaries_type):
            if b != BoundaryType.CYCLIC:
                vec[i] = dx[i]
                continue
            l = boundaries.edges[i]
            dxs = np.array([dx[i], dx[i] + l, dx[i] - l])
            vec[i] = dxs[np.argmin(dxs ** 2)]  # find shorter path through B.D.
        return vec

    @staticmethod
    def cyclic_dist(boundaries, sphere1, sphere2):
        dx = np.array(sphere1.center) - sphere2.center  # direct vector
        dsq = 0
        for i, b in enumerate(boundaries.boundaries_type):
            if b != BoundaryType.CYCLIC:
                dsq += dx[i] ** 2
                continue
            L = boundaries.edges[i]
            dsq += min(dx[i] ** 2, (dx[i] + L) ** 2, (dx[i] - L) ** 2)  # find shorter path through B.D.
        return np.sqrt(dsq)

    @staticmethod
    def overlap(sphere1: Sphere, sphere2: Sphere, boundaries: CubeBoundaries):
        """
        Test of two d-dimensional Sphere objects overlap
        :type sphere1: Sphere
        :type sphere2: Sphere
        :type boundaries: CubeBoundaries
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
    def spheres_overlap(spheres, boundaries: CubeBoundaries):
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
                if sphere not in self.spheres:
                    self.spheres.append(sphere)
        else:
            assert type(new_spheres) == Sphere
            if new_spheres not in self.spheres:
                self.spheres.append(new_spheres)

    def remove_sphere(self, spheres_to_remove):
        """
        Delete spheres from the cell
        :param spheres_to_remove: the Sphere objects to be removed (pointers!)
        """
        if type(spheres_to_remove) == list:
            for sphere in spheres_to_remove: self.spheres.remove(sphere)
        else:
            assert type(spheres_to_remove) == Sphere
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

    def random_generate_spheres(self, n_spheres, rads, extra_edges=[]):
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
                center = np.array(self.site) + [random.random() * e for e in self.edges]
                if len(extra_edges) > 0:
                    center = [c for c in center] + [r + epsilon + random.random() * (e - 2 * (r + epsilon)) for e in
                                                    extra_edges]
                    # assumes for now extra edge is rigid wall and so generate in the allowed locations
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
        :type boundaries: CubeBoundaries
        :param dim: dimension of the array of cells. Doesn't have to be dimension of a single sphere.
        :param cells: list of cells defining the array, optional.
        """
        self.dim = dim
        self.cells = cells
        self.boundaries = boundaries
        self.n_rows = len(cells)
        self.n_columns = len(cells[0])

    @property
    def all_cells(self):
        return [c for c in np.reshape(self.cells, -1)]

    @property
    def all_spheres(self):
        """
        :return: list of Sphere objects of all the spheres in the array
        """
        spheres = []
        for cell in self.all_cells:
            if cell == []: continue
            for sphere in cell.spheres:
                spheres.append(sphere)
        return spheres

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
        x_cyclic = self.boundaries.boundaries_type[0] == BoundaryType.CYCLIC
        y_cyclic = self.boundaries.boundaries_type[1] == BoundaryType.CYCLIC
        both_cyclic = x_cyclic and y_cyclic
        cells = [[Cell((), []) for _ in range(n_columns + 2)] for _ in range(n_rows + 2)]
        for i in range(n_rows):  # 1 < (i + 1, j+1) < n
            for j in range(n_columns):
                cells[i + 1][j + 1] = self.cells[i][j]
        l_x = self.boundaries.edges[0]
        for i in range(n_rows):  # 1 < i + 1 < n
            c0 = copy.deepcopy(self.cells[i][n_columns - 1])
            c0.transform(c0.site + np.array([-l_x, 0]))
            c1 = copy.deepcopy(self.cells[i][0])
            c1.transform(c1.site + np.array([l_x, 0]))
            if x_cyclic:
                cells[i + 1][0] = c0
                cells[i + 1][n_columns + 1] = c1
            else:
                cells[i + 1][0] = Cell(c0.site, c0.edges)
                cells[i + 1][n_columns + 1] = Cell(c1.site, c1.edges)
        l_y = self.boundaries.edges[1]
        for j in range(n_columns):  # 1 < j + 1 < n
            c0 = copy.deepcopy(self.cells[n_rows - 1][j])
            c0.transform(c0.site + np.array([0, -l_y]))
            c1 = copy.deepcopy(self.cells[0][j])
            c1.transform(c1.site + np.array([0, l_y]))
            if y_cyclic:
                cells[0][j + 1] = c0
                cells[n_rows + 1][j + 1] = c1
            else:
                cells[0][j + 1] = Cell(c0.site, c0.edges)
                cells[n_rows + 1][j + 1] = Cell(c1.site, c1.edges)
        if both_cyclic:
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
        else:
            c = copy.deepcopy(self.cells[n_rows - 1][n_columns - 1])
            c.transform(c.site + np.array([-l_x, -l_y]))
            cells[0][0] = Cell(c.site, c.edges)
            c = copy.deepcopy(self.cells[0][0])
            c.transform(c.site + np.array([l_x, l_y]))
            cells[n_rows + 1][n_columns + 1] = Cell(c.site, c.edges)
            c = copy.deepcopy(self.cells[0][n_columns - 1])
            c.transform(c.site + np.array([-l_x, l_y]))
            cells[n_rows + 1][0] = Cell(c.site, c.edges)
            c = copy.deepcopy(self.cells[n_rows - 1][0])
            c.transform(c.site + np.array([l_x, -l_y]))
            cells[0][n_columns + 1] = Cell(c.site, c.edges)
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
        if self.all_cells == []: return True
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
                if self.boundaries.dim == 3:
                    for sphere in cell.spheres:
                        c_z = sphere.center[2]
                        r = sphere.rad
                        if self.boundaries.boundaries_type[2] == BoundaryType.WALL and \
                                (c_z - r < -epsilon or c_z + r > self.boundaries.edges[2] + epsilon):
                            return False
                if (j == n_columns - 1 or j == 0) and self.boundaries.boundaries_type[0] == BoundaryType.WALL:
                    for sphere in cell.spheres:
                        c_x = sphere.center[0]
                        r = sphere.rad
                        if c_x - r < -epsilon or c_x + r > self.boundaries.edges[0] + epsilon:
                            return False
                if (i == n_rows - 1 or i == 0) and self.boundaries.boundaries_type[1] == BoundaryType.WALL:
                    for sphere in cell.spheres:
                        c_y = sphere.center[1]
                        r = sphere.rad
                        if c_y - r < -epsilon or c_y + r > self.boundaries.edges[1] + epsilon:
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

    def random_generate_spheres(self, n_spheres_per_cell, rad, extra_edges=[]):
        if type(rad) != list: rad = n_spheres_per_cell * [rad]
        while True:
            for cell in self.all_cells:
                cell.random_generate_spheres(n_spheres_per_cell, rad, extra_edges)
            if self.legal_configuration():
                return

    def generate_spheres_in_cubic_structure(self, n_spheres_per_cell, rad, extra_edges=[]):
        if type(rad) != list: rad = n_spheres_per_cell * [rad]
        for cell in self.all_cells:
            dx, dy = epsilon, epsilon
            x0, y0 = cell.site
            max_r = 0
            for i in range(n_spheres_per_cell):
                r = rad[i]
                if r > max_r: max_r = r
                center = (x0 + dx + r, y0 + dy + r)
                if len(extra_edges) > 0:
                    center = [c for c in center] + [r + random.random() * (e - 2 * r) for e in extra_edges]
                cell.append(Sphere(center, r))
                dx += 2 * r + epsilon
                if (i < n_spheres_per_cell - 1) and (dx + 2 * rad[i + 1] > cell.edges[0]):
                    dx = 0
                    dy += 2 * max_r + epsilon
                    max_r = 0
                if (i < n_spheres_per_cell - 1) and (dy + 2 * rad[i + 1] > cell.edges[1]):
                    break
        assert self.legal_configuration()

    def append_sphere(self, spheres):
        if type(spheres) != list:
            assert type(spheres) == Sphere
            spheres = [spheres]
        cells = []
        for sphere in spheres:
            sp_added_to_cell = False
            for c in self.all_cells:
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
                spheres.append(Sphere((xj, yi), rad))
        return spheres

    def translate(self, vec):
        """
        Move all spheres
        :param vec: vector of translation.
        :return: transfered spheres, after removing all spheres from the array of cells
        """
        transferred_spheres = []
        vec = np.array(vec)
        for c in self.all_cells:
            for s in c.spheres:
                for i in range(min(len(s.center), len(vec))):
                    s.center[i] += vec[i]
                transferred_spheres.append(s)
        for c in self.all_cells:
            c.spheres = []
        # self.append_sphere(transferred_spheres)
        return transferred_spheres
