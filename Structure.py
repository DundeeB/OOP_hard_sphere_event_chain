import numpy as np
import random
import copy
from enum import Enum
epsilon = 1e-6


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

    def sphere_dist(self, other_sphere):
        """
        :type other_sphere: Sphere
        :return: distance between two spheres, without boundary conditions
        """
        return np.linalg.norm(np.array(self.center) - other_sphere.center)

    @staticmethod
    def overlap(sphere1, sphere2):
        """
        Test of two d-dimensional Sphere objects over lap
        :type sphere1: Sphere
        :type sphere2: Sphere
        :return: True if they overlap
        """
        delta = sphere1.sphere_dist(sphere2) - (sphere1.rad + sphere2.rad)
        if delta < -epsilon:
            return True
        else:
            if delta > 0:
                return False
            else:
                Warning("Spheres are epsilon close to each other, seperating them")
                dr_hat = sphere1.center - np.array(sphere2.center)
                dr_hat = dr_hat / (np.linalg.norm(dr_hat))
                sphere1.center = sphere1.center + epsilon*dr_hat

    @staticmethod
    def spheres_overlap(spheres):
        """
        :type spheres: list
        :param spheres: list of spheres to check overlap between all couples
        :return:
        """
        for i in range(len(spheres)):
            for j in range(i):
                if Sphere.overlap(spheres[i], spheres[j]):
                    return True
        return False

    def box_it(self, boundaries):
        """
        Put the sphere inside the boundaries of the simulation, usefull for cyclic boundary conditions
        :type boundaries: CubeBoundaries
        """
        self.center = [x % e for x, e in zip(self.center, boundaries.edges)]

    def perform_step(self, step):
        """
        :type step: Step
        :param step: step to be perform
        :return:
        """
        self.center = self.center + np.array(step.v_hat)*step.current_step
        self.box_it(step.boundaries)

    def systems_length_in_v_direction(self, v_hat, boundaries):
        """
        :type boundaries: CubeBoundaries
        """
        lx = boundaries.edges[0]
        ly = boundaries.edges[1]
        vx = v_hat[0]
        vy = v_hat[1]
        x_hat = (1, 0)
        y_hat = (0, 1)
        r_c = np.array((self.center[0], self.center[1]))
        if vx != 0 and vy != 0:
            dist_upper_boundary = min(abs((ly - np.dot(r_c, y_hat)) / vy),
                                      abs((lx - np.dot(r_c, x_hat)) / vx))
            dist_bottom_boundary = min(abs((0 - np.dot(r_c, y_hat)) / vy),
                                       abs((0 - np.dot(r_c, x_hat)) / vx))
        if vx != 0  and vy ==0:
            dist_upper_boundary = abs((lx - np.dot(r_c, x_hat)) / vx)
            dist_bottom_boundary = abs((0 - np.dot(r_c, x_hat)) / vx)
        if vx == 0 and vy != 0:
            dist_upper_boundary = abs((ly - np.dot(r_c, y_hat)) / vy)
            dist_bottom_boundary = abs((0 - np.dot(r_c, y_hat)) / vy)
        return dist_bottom_boundary + dist_upper_boundary

    def trajectory(self, t, v_hat, boundaries):
        """
        Solve for the trajectory from the starting point self.center, assuming cyclic boundary conditions
        :param t: length of step
        :param v_hat: direction of step
        :type boundaries: CubeBoundaries
        :return:
        """
        r = self.center+np.array(v_hat)*t/np.linalg.norm(v_hat)
        r = [x % e for x, e in zip(r, boundaries.edges)]
        return np.array(r)

    def trajectories_braked_to_lines(self, total_step, v_hat, boundaries):
        """
        return list [p1, p2, ..., pn] of points for which p1 and direction v_hat defines the trajectories. p1 is the
        starting point and last pn is the last point. All the points in between are at the bottom or to the left of the
        simulation
        :param total_step: total step left to be carried out
        :param v_hat: direction of step
        :type boundaries: CubeBoundaries
        :return: list [p1, p2, ..., pn] of points for which p1 and direction v_hat defines the trajectories
        """
        len_v = self.systems_length_in_v_direction(v_hat, boundaries)
        first_step, _ = Metric.dist_to_boundary_without_r(self, total_step, v_hat, boundaries)
        if first_step == float('inf'):
            return [self.center, self.trajectory(total_step, v_hat, boundaries)], \
                   [0, total_step]
        ts = [first_step + len_v * k for k in
              range(int(np.floor(((total_step - first_step) / len_v)))+1)]
        if ts[-1] != total_step: ts.append(total_step)
        ps = [self.center] + [self.trajectory(t, v_hat, boundaries) for t in ts]
        ps = [np.array(p) for p in ps]
        return ps, [0] + ts


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
        assert (len(edges) == len(boundaries_type))
        for bound in boundaries_type:
            assert(type(bound) == BoundaryType)
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
    def walls(self):
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
    def walls_type(self):
        """
        :return: list of boundary conditions, the i'th boundary condition is for i'th wall in self.walls
        """
        bc0 = self.boundaries_type[0]
        if self.dim == 1: return 2*[bc0]
        bc1 = self.boundaries_type[1]
        if self.dim == 2:
            return [bc1, bc0, bc0, bc1]
        else:  # dim=3
            bc2 = self.boundaries_type[2]
            # xy yz xz xy yz xz
            return 2*[bc2, bc0, bc1]

    @staticmethod
    def vertical_step_to_wall(wall, point):
        """
        For a wall=[v0,v1,...], which is a plain going through all the vertices [v0,v1...],
        Return the vertical to the plain vector, with length the distance to the point
        :param wall: list of vertices, which the define the wall which is the plain going through them
        :param point: the specified point to get the vertical step to plain from
        :return: the smallest vector v s.t. v+point is on the plain of the wall
        """
        for w in wall:
            assert(len(w) == len(point))
        d = len(point)
        p = np.array(point)
        v = [np.array(w) for w in wall]
        if d == 1:
            return v[0] - p
        if d == 2:
            t = -np.dot(v[0]-p, v[1]-v[0])/np.dot(v[1]-v[0], v[1]-v[0])
            return v[0] - p + t*(v[1]-v[0])
        if d == 3:
            # assume for now that edges are vertical so calculation is easier
            assert(np.dot(v[2] - v[0], v[1] - v[0]) < epsilon)
            t = -np.dot(v[0] - p, v[1] - v[0]) / np.dot(v[1] - v[0], v[1] - v[0])
            s = -np.dot(v[0] - p, v[2] - v[0]) / np.dot(v[2] - v[0], v[2] - v[0])
            return v[0] - p + t*(v[1]-v[0]) + s*(v[2]-v[0])

    @staticmethod
    def flip_v_hat_wall_part(wall, sphere, v_hat):
        """
        Next to rigid wall boundary condition, we would want v_hat to flip direction
        :param wall: list of points, definig the wall's plane
        :type sphere: Sphere
        :param v_hat: current direction of step
        :return: flipped direction of  step, opposite to wall
        """
        n_hat = CubeBoundaries.vertical_step_to_wall(wall, sphere.center)
        n_hat = np.array(n_hat)/np.linalg.norm(n_hat)
        return v_hat-2*np.dot(v_hat, n_hat)*n_hat

    def sphere_dist(self, sphere1, sphere2):
        """
        :type sphere1: Sphere
        :type sphere2: Sphere
        :return: distance between the two sphere, but including CYCLIC boundary conditions
        """
        direct = sphere1.sphere_dist(sphere2)
        dist = direct
        cloned_sphere = copy.deepcopy(sphere1)
        l_x = self.edges[0]
        l_y = self.edges[1]
        if self.dim == 3 and self.boundaries_type[2] == BoundaryType.CYCLIC: raise Exception('z wall CYCLIC not supported')
        for bound_vec, boundary_type in zip([[l_x, 0], [0, l_y], [-l_x, 0], [0, -l_y]], 2*self.boundaries_type[0:2]):
            if boundary_type != BoundaryType.CYCLIC: continue
            if self.dim == 3: bound_vec = [b for b in bound_vec] + [0]
            cloned_sphere.center = sphere1.center + np.array(bound_vec)
            new_dist = cloned_sphere.sphere_dist(sphere2)
            if new_dist < dist: dist = new_dist
        return dist

    def overlap(self, sphere1, sphere2):
        """
        :type sphere1: Sphere
        :type sphere2: Sphere
        :return: True if sphere1 and sphere2 are overlaping, even through CYCLIC boundary condition
        """
        return (self.sphere_dist(sphere1, sphere2) <= sphere1.rad + sphere2.rad)

    def spheres_overlap(self, spheres):
        """
        :type spheres: list
        :param spheres: list of spheres to check overlap between all couples
        :return: True if there are two overlaping spheres, even if they overlap through boundary condition
        """
        for i in range(len(spheres)):
            for j in range(i):
                if self.overlap(spheres[i],spheres[j]):
                    return True
        return False


class Metric:

    @staticmethod
    def dist_to_boundary(sphere, total_step, v_hat, boundaries):
        """
        Figures out the distance for the next WALL boundary (ignoring CYCLIC boundaries)
        :type sphere: Sphere
        :param v_hat: direction of step
        :param total_step: magnitude of the step to be carried out
        :type boundaries: CubeBoundaries
        :return: the minimal distance to the wall, and the wall.
        If there is no wall in a distance l, dist is inf and wall=[]
        """
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)
        pos = np.array(sphere.center)
        r = sphere.rad
        min_dist_to_wall = float('inf')
        closest_wall = []
        for wall, BC_type in zip(boundaries.walls, boundaries.walls_type):
            if BC_type != BoundaryType.WALL: continue
            u = CubeBoundaries.vertical_step_to_wall(wall, pos)
            u = u - r*u/np.linalg.norm(u)  # shift the wall closer by r
            if np.dot(u, v_hat) < 0:
                continue
            v = np.dot(u, u)/np.dot(u, v_hat)
            if v < min_dist_to_wall and v <= total_step:
                min_dist_to_wall = v
                closest_wall = wall
        return min_dist_to_wall, closest_wall

    @staticmethod
    def dist_to_boundary_without_r(sphere, total_step, v_hat, boundaries):
        """
        Returns the distance to the boundary without taking into account the sphere radius, +epsilon so after box_it it
        will give the bottom or the left boundary point. Also ignore whether its a wall or a cyclic boundary
        :type sphere: Sphere
        :param total_step: total step left to be performed
        :param v_hat: direction of step
        :type boundaries: CubeBoundaries
        """
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)
        pos = np.array(sphere.center)
        min_dist_to_wall = float('inf')
        closest_wall = []
        for wall in boundaries.walls:
            u = CubeBoundaries.vertical_step_to_wall(wall, pos)
            if np.dot(u, v_hat) <= 0:
                continue
            v = np.dot(u, u) / np.dot(u, v_hat)
            if v < min_dist_to_wall and v <= total_step:
                min_dist_to_wall = v
                closest_wall = wall
        return min_dist_to_wall + epsilon, closest_wall

    @staticmethod
    def dist_to_collision(sphere1, sphere2, total_step, v_hat, boundaries):
        """
        Distance sphere1 would need to go in v_hat direction in order to collide with sphere2
        It is not implemented in the most efficient way, as each time two spheres are compared we need to run over all
        the first sphere's trajectory. It is not important, because in most cases a sphere would cross at most one
        boundary condition, so we can allow ourselves to have a less efficient implementation.
        :param sphere1: sphere about to move
        :type sphere1: Sphere
        :param sphere2: potential for collision
        :type sphere2: Sphere
        :param total_step: maximal step size. If dist for collision > total_step then dist_to_collision-->infty
        :param v_hat: direction in which sphere1 is to move
        :param boundaries: boundaries for the case some of them are cyclic boundary conditinos
        :type boundaries: CubeBoundaries
        :return: distance for collision if the move is allowed, infty if move can not lead to collision
        """
        assert(not Sphere.overlap(sphere1, sphere2))
        d = sphere1.rad + sphere2.rad
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)
        ps, ts = sphere1.trajectories_braked_to_lines(total_step, v_hat, boundaries)
        pos_b = sphere2.center
        for pos_a, t in zip(ps, ts):
            dx = np.array(pos_b) - np.array(pos_a)
            dx_len = np.linalg.norm(dx)
            dx_dot_v = np.dot(dx, v_hat)
            if dx_dot_v <= 0:
                continue
            if dx_len - d <= epsilon:
                # new pos_a is already overlapping, meaning the sphere is leaking through the boundary condition
                go_back = -dx_dot_v + np.sqrt(dx_dot_v**2 + d**2 - dx_len**2)
                return t - go_back
            discriminant = dx_dot_v ** 2 + d ** 2 - np.linalg.norm(dx) ** 2
            if discriminant > 0:  # found a solution!
                dist: float = t + dx_dot_v - np.sqrt(discriminant)
                if dist <= total_step: return dist
        return float('inf')


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

    def add_spheres(self, new_spheres):
        """
        Add spheres to the cell
        :param new_spheres: list of Sphere objects to be added to the cell
        """
        try:
            for sphere in new_spheres: self.spheres.append(sphere)
        except TypeError as single_sphere_exception:
            assert(single_sphere_exception.args[0], '\'Sphere\' object is not iterable')
            self.spheres.append(new_spheres)

    def remove_sphere(self, spheres_to_remove):
        """
        Delete spheres from the cell
        :param spheres_to_remove: the Sphere objects to be removed (pointers!)
        """
        try:
            for sphere in spheres_to_remove: self.spheres.remove(sphere)
        except TypeError as single_sphere_exception:
            assert(single_sphere_exception.args[0], '\'Sphere\' object is not iterable')
            self.spheres.remove(spheres_to_remove)

    def sphere_in_cell(self, sphere):
        """
        A Sphere object should be in a cell if its center is inside the physical d-dimension
        cube of the cell. Notice in the case dim(cell)!=dim(sphere), a sphere is considered
        inside a cell if its dim(cell) first components are inside the cell.
        :param sphere: the sphere to check whether should be considered in the new cell
        :type sphere: Sphere
        :return: True if the sphere should be considered inside the cell, false otherwise
        """
        for i in range(len(self.site)):
            x_sphere, x_site, edge = sphere.center[i], self.site[i], self.edges[i]
            #Notice this implementation instead of for x_... in zip() is for the case dim(sphere)!=dim(cell)
            if x_sphere <= x_site or x_sphere > x_site + edge:
                return False
        return True

    @property
    def dim(self):
        return len(self.site)

    def random_generate_spheres(self, n_spheres, rads, extra_edges=[]):
        """
        Generate n spheres inside the cell. If there are spheres in the cell already,
         it deletes the existing spheres. The algorithm is to randomaly
         generate their centers and check for overlap. In order to save time,
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
                center = np.array(self.site) + [random.random()*e for e in self.edges]
                if len(extra_edges) > 0:
                    center = [c for c in center] + [r + random.random()*(e - 2*r) for e in extra_edges]
                    #assumes for now extra edge is rigid wall and so generate in the allowed locations
                spheres.append(Sphere(center, rads[i]))
            if not Sphere.spheres_overlap(spheres):
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
                dx = [dx_ for dx_ in dx] + [0 for _ in range(sphere.dim-len(dx))]
                dx = np.array(dx)
            sphere.center = sphere.center + dx


class ArrayOfCells:

    def __init__(self, dim, boundaries, cells=[]):
        """
        :type boundaries: CubeBoundaries
        :param dim: dimension of the array of cells. Doesn't have to be dimension of a single sphere.
        :param cells: list of cells defining the array, optional.
        """
        self.dim = dim
        self.cells = cells
        self.boundaries = boundaries

    @property
    def all_spheres(self):
        """
        :return: list of Sphere objects of all the spheres in the array
        """
        spheres = []
        for cell in np.reshape(self.cells, -1):
            if cell == []: continue
            for sphere in cell.spheres:
                spheres.append(sphere)
        return spheres

    @property
    def all_centers(self):
        """
        :return: list of all the centers (d-dimension vectors) of all the Sphere objects in the simulation.
        """
        centers = []
        for cell in np.reshape(self.cells, -1):
            for sphere in cell.spheres:
                centers.append(sphere.center)
        return centers

    @property
    def all_cells(self):
        return [c for c in np.reshape(self.cells, -1)]

    def overlap_2_cells(self, cell1, cell2):
        """
        Checks if the spheres in cell1 and cell2 are overlapping with each other. Does not check inside cell.
        :type cell1: Cell
        :type cell2: Cell
        :return: True if one of the spheres in cell1 overlap with one of the spheres in cell2
        """
        if cell1 == [] or cell2 == []:
            return False
        spheres1 = cell1.spheres
        spheres2 = cell2.spheres
        for sphere1 in spheres1:
            for sphere2 in spheres2:
                if self.boundaries.overlap(sphere1, sphere2):
                    return True
        return False

    def cushioning_array_for_boundary_cond(self):
        """
        :return: array of cells of length n_row+2 n_column+2, for any CYCLIC boundary condition it is surrounded by the
        cells in the other end
        """
        if self.dim != 2: raise Exception("dim!=2 not implemented yet")
        n_rows = len(self.cells)
        n_columns = len(self.cells[0])
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

    def legal_configuration(self):
        """
        :return: True if there are no overlapping spheres in the configuration
        """
        if self.cells == []: return True
        d = self.dim
        if d != 2:
            raise (Exception('Only d=2 supported!'))
        cushioned_cells = self.cushioning_array_for_boundary_cond().cells
        n_rows = len(cushioned_cells) - 2
        for i in range(1, n_rows + 1):
            n_columns = len(cushioned_cells[i]) - 2
            for j in range(1, n_columns + 1):
                cell = cushioned_cells[i][j]
                if Sphere.spheres_overlap(cell.spheres):
                    return False
                if self.boundaries.dim == 3:
                    for sphere in cell.spheres:
                        c_z = sphere.center[2]
                        r = sphere.rad
                        if self.boundaries.boundaries_type[2] == BoundaryType.WALL and \
                            (c_z - r < 0 or c_z + r > self.boundaries.edges[2]):
                            return False
                if (j == n_columns or j == 1) and self.boundaries.boundaries_type[0] == BoundaryType.WALL:
                    for sphere in cell.spheres:
                        c_x = sphere.center[0]
                        r = sphere.rad
                        if c_x - r < 0 or c_x + r > self.boundaries.edges[0]:
                            return False
                if (i == n_rows or i == 1) and self.boundaries.boundaries_type[1] == BoundaryType.WALL:
                    for sphere in cell.spheres:
                        c_y = sphere.center[1]
                        r = sphere.rad
                        if c_y - r < 0 or c_y + r > self.boundaries.edges[1]:
                            return False
                neighbors = [cushioned_cells[i + 1][j - 1], cushioned_cells[i + 1][j],
                             cushioned_cells[i + 1][j + 1], cushioned_cells[i][j + 1]]
                for neighbor in neighbors:
                    if self.overlap_2_cells(cell, neighbor):
                        return False
        return True

    def cell_from_ind(self, ind):
        cell = self.cells
        for i in ind: cell = cell[i]
        return cell

    def random_generate_spheres(self, n_spheres_per_cell, rad, extra_edges=[]):
        if type(rad) != list: rad = n_spheres_per_cell*[rad]
        while True:
            for cell in self.all_cells:
                cell.random_generate_spheres(n_spheres_per_cell, rad, extra_edges)
            if self.legal_configuration():
                return
