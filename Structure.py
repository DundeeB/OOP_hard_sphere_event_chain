import numpy as np
import random
from enum import Enum
epsilon = 1e-4


class Sphere:

    def __init__(self, center, rad):
        """
        Create new sphere object
        :param center: d dimension of sphere center
        :param rad: scalar - radius of sphere
        """
        self.center = center
        self.rad = rad

    def dim(self):
        """
        :return: d - the dimension of the Sphere instance
        """
        return len(self.center)

    @staticmethod
    def overlap(sphere1, sphere2):
        """
        Test of two d-dimensional Sphere objects over lap
        :type sphere1: Sphere
        :type sphere2: Sphere
        :return: True if they overlap
        """
        return np.linalg.norm(sphere1.center-sphere2.center) < sphere1.rad + sphere2.rad

    @staticmethod
    def spheres_overlap(spheres):
        """
        :type spheres: list
        :param spheres: list of spheres to check overlap between all couples
        :return:
        """
        for i in range(len(spheres)):
            for j in range(i):
                if Sphere.overlap(spheres[i],spheres[j]):
                    return True
        return False

    def box_it(self, boundaries):
        """
        Put the sphere inside the boundaries of the simulation, usefull for cyclic boundary conditions
        :type boundaries: CubeBoundaries
        """
        self.center = [x % e for x, e in zip(self.center, boundaries.edges)]

    def perform_step(self, step, boundaries):
        """
        :type step: Step
        :param step: step to be perform
        :type boundaries: CubeBoundaries
        :param boundaries: boundaries of the simulation, needed to know for the case of cyclic boundary condition
        :return:
        """
        self.center = self.center + np.array(step.v_hat)*step.current_step
        self.box_it(boundaries)

    def systems_length_in_v_direction(self, v_hat, boundaries):
        """
        :type v_hat: list
        :type boundaries: CubeBoundaries
        """
        lx = boundaries.edges[0]
        ly = boundaries.edges[1]
        vx = np.dot(v_hat, [1, 0])
        vy = np.dot(v_hat, [0, 1])
        if vx != 0 and vy != 0:
            dist_upper_boundary = min(abs((ly - np.dot(self.center, [0, 1])) / vy),
                                      abs((lx - np.dot(self.center, [1, 0])) / vx))
            dist_bottom_boundary = min(abs((0 - np.dot(self.center, [0, 1])) / vy),
                                       abs((0 - np.dot(self.center, [1, 0])) / vx))
        if vx != 0  and vy ==0:
            dist_upper_boundary = abs((lx - np.dot(self.center, [1, 0])) / vx)
            dist_bottom_boundary = abs((0 - np.dot(self.center, [1, 0])) / vx)
        if vx == 0 and vy != 0:
            dist_upper_boundary = abs((ly - np.dot(self.center, [0, 1])) / vy)
            dist_bottom_boundary = abs((0 - np.dot(self.center, [0, 1])) / vy)
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
        return r

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
        first_step = Metric.dist_to_boundary_without_r(self, total_step, v_hat, boundaries)
        ts = [first_step + len_v * k for k in
              range(int(np.floor(((total_step - first_step) / len_v))))]
        if ts[-1] != total_step: ts.append(total_step)
        ps = [self.center] + [self.trajectory(t, v_hat, boundaries) for t in ts]
        return ps


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
        for sphere in new_spheres: self.spheres.append(sphere)

    def remove_sphere(self, spheres_to_remove):
        """
        Delete spheres from the cell
        :param spheres_to_remove: the Sphere objects to be removed (pointers!)
        """
        for sphere in spheres_to_remove: self.spheres.remove(sphere)

    def should_sphere_be_in_cell(self, sphere):
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
            if x_sphere < x_site or x_sphere > x_site + edge:
                return False
        return True

    def dim(self):
        return len(self.site)

    def random_generate_spheres(self, n_spheres, rads, extra_edges=[]):
        """
        Generate n spheres inside the cell. If there are spheres in the cell already,
         it deletes the exisiting spheres. The algorithm is to randomaly
         generate their centers and check for overlap. In order to save time,
         they are generated with a rad distance from the end of the cell, which
         might cause problems in the future
        :param n_spheres: number of spheres to be generated
        :param rads: rads of the different spheres
        :param extra_edges: if dim_spheres>dim(cell), there should be constraint on the extra
         dimension size. For 2d cells with 3d spheres confined between two walls, extra_edges=[h]
        """
        while True: # do-while python implementation
            spheres = []
            for i in range(n_spheres):
                r=rads[i]
                center = np.array(self.site) + [r + random.random()*(e-r) for e in self.edges]
                if len(extra_edges)>0:
                    center = [c for c in center] + [r + random.random()*(e-r) for e in extra_edges]
                spheres.append(Sphere(center,rads[i]))
            if not Sphere.spheres_overlap(spheres):
                break
        self.spheres = spheres

    def transform(self, new_site):
        """
        Transform the cell and all its spheres to the new_site.
        Might be usefull for boundary conditinos implementation
        :param new_site: new_site for the cell to transform to
        """
        dx = np.array(new_site - self.site)
        self.site = new_site
        for sphere in self.spheres:
            sphere.center = sphere.center + dx


class BoundaryType(Enum):
    WALL = "RigidWall"
    CYCLIC = "CyclicBoundaryConditions"


class CubeBoundaries:

    def __init__(self, edges, boundaries_type):
        """
        Create new boundaries for the simulation
        :param edges: list of edge per dimension for the simulation
        """
        self.edges = edges
        self.boundaries_type = boundaries_type
        self.dim = len(edges)

    def get_vertices(self):
        if self.dim == 1:
            return [(0,), (self.edges[0],)]
        if self.dim == 2:
            return [(0, 0), (0, self.edges[1]), (self.edges[0], 0),
                    (self.edges[0], self.edges[1])]
        else:  # dim==3
            e0 = self.edges[0]
            e1 = self.edges[1]
            e2 = self.edges[2]
            return [(0, 0, 0), (e0, 0, 0), (0, e1, 0), (0, 0, e2),
                    (0, e1, e2), (e0, 0, e2), (e0, e1, 0), (e0, e1, e2)]

    def get_walls(self):
        vs = self.get_vertices()
        if self.dim == 1:
            return [(vs[0],), (vs[1],)]
        if self.dim == 2:
            return [(vs[0], vs[1]), (vs[0], vs[2]),
                    (vs[1], vs[3]), (vs[2], vs[3])]
        else:  # dim==3
            return [(vs[0], vs[2], vs[1], vs[6]),
                    (vs[0], vs[2], vs[3], vs[4]),
                    (vs[0], vs[3], vs[1], vs[5]),
                    (vs[4], vs[7], vs[3], vs[5]),
                    (vs[1], vs[5], vs[6], vs[7]),
                    (vs[2], vs[4], vs[6], vs[7])]

    @staticmethod
    def vertical_step_to_wall(wall, point):
        """
        For a wall=[v0,v1,...], which is a plain going through all the vertices [v0,v1...],
        Return the vertical to the plain vector toward the point
        :param wall: list of vertices, which the define the wall which is the plain going through them
        :param point: the specified point to get the vertical step to plain from
        :return: the smallest vector v s.t. v+point is on the plain of the wall
        """
        assert(len(wall) == len(point))
        d = len(wall)
        p = np.array(point)
        v = [np.array(w) for w in wall]
        if d == 1:
            return v[0] - p
        if d == 2:
            t = -np.dot(v[0]-p, v[1]-v[0])/np.dot(v[1]-v[0], v[1]-v[0])
            return v[0] - p + t*(v[1]-v[0])
        if d == 3:
            # assume for now that edges are vertical so calculation is easier
            assert(np.dot(v[2]-v[0],v[1]-v[0]) < epsilon)
            t = -np.dot(v[0]-p, v[1]-v[0])/np.dot(v[1]-v[0], v[1]-v[0])
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
        return v_hat-2*np.dot(v_hat,n_hat)*n_hat


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

    def get_all_spheres(self):
        """
        :return: list of Sphere objects of all the spheres in the array
        """
        spheres = []
        for cell in np.reshape(self.cells,-1):
            for sphere in cell.spheres:
                spheres.append(sphere)
        return spheres

    def get_all_centers(self):
        """
        :return: list of all the centers (d-dimension vectors) of all the Sphere objects in the simulation.
        """
        centers = []
        for cell in np.reshape(self.cells,-1):
            for sphere in cell.spheres:
                centers.append(sphere.center)
        return centers

    @staticmethod
    def overlap_2_cells(cell1, cell2):
        """
        Checks if the spheres in cell1 and cell2 are overlapping with each other. Does not check inside cell
        :type cell1: Cell
        :type cell2: Cell
        :return: True if one of the spheres in cell1 overlap with one of the spheres in cell2
        """
        spheres1 = cell1.spheres
        spheres2 = cell2.spheres
        for sphere1 in spheres1:
            for sphere2 in spheres2:
                if Sphere.overlap(sphere1,sphere2):
                    return True
        return False

    def legal_configuration(self):
        """
        :return: True if there are no overlapping spheres in the configuration
        """
        if self.cells == []: return True
        d = self.cells[0].dim()
        if d != 2 and d != 3:
            raise (Exception('Only d=2 or d=3 supported!'))
        for i in range(len(self.cells) - 1):
            for j in range(len(self.cells[i]) - 1):
                if d == 2:
                    cell = self.cells[i][j]
                    neighbors = [self.cells[i + 1][j], self.cells[i][j + 1],
                                 self.cells[i + 1][j + 1]]
                    if Sphere.spheres_overlap(cell.spheres):
                        return False
                    for neighbor in neighbors:
                        if ArrayOfCells.overlap_2_cells(cell, neighbor):
                            return False

                if d == 3:
                    for k in range(len(self.cells[i][j]) - 1):
                        cell = self.cells[i][j][k]
                        neighbors = [self.cells[i + 1][j][k], self.cells[i][j + 1][k],
                                     self.cells[i][j][k + 1], self.cells[i + 1][j + 1][k],
                                     self.cells[i + 1][j][k + 1], self.cells[i][j + 1][k + 1],
                                     self.cells[i + 1][j + 1][k + 1]]
                        if Sphere.spheres_overlap(cell.spheres):
                            return False
                        for neighbor in neighbors:
                            if ArrayOfCells.overlap_2_cells(cell, neighbor):
                                return False

        return True

    def cell_from_ind(self, ind):
        cell = self.cells
        for i in ind: cell = cell[i]
        return cell


class Metric:

    @staticmethod
    def dist_to_boundary(sphere, total_step, v_hat, boundaries):
        """
        Figures out the distance for the next boundary
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
        for wall, BC_type in zip(boundaries.get_walls(), boundaries.boundaries_type):
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
        returns the distance to the boundary without taking into account the sphere radius, +epsilon so after box_it it
        will give the bottom or the left boundary point. Also ignore whether its a wall or a cyclic boundary
        :type sphere: Sphere
        :param total_step: total step left to be performed
        :param v_hat: direction of step
        :type boundaries: CubeBoundaries
        """
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)
        pos = np.array(sphere.center)
        min_dist_to_wall = float('inf')
        for wall in boundaries.get_walls():
            u = CubeBoundaries.vertical_step_to_wall(wall, pos)
            if np.dot(u, v_hat) < 0:
                continue
            v = np.dot(u, u) / np.dot(u, v_hat)
            if v < min_dist_to_wall and v <= total_step:
                min_dist_to_wall = v
        return min_dist_to_wall + epsilon

    @staticmethod
    def dist_to_collision(sphere1, sphere2, total_step, v_hat, boundaries):
        """
        Distance sphere1 would need to go in v_hat direction in order to collide with sphere2
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
        pos_a = sphere1.center
        pos_b = sphere2.center
        d = sphere1.rad + sphere2.rad
        v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
        dx = np.array(pos_b) - np.array(pos_a)
        dx_dot_v = np.dot(dx, v_hat)
        if dx_dot_v <= 0:
            # cyclic boundary condition are relevant
            total_step = sphere1.systems_length_in_v_direction(v_hat, boundaries)
            pos_a = sphere1.center-total_step*v_hat
            dx = np.array(pos_b) - np.array(pos_a)
            dx_dot_v = np.dot(dx, v_hat)

        if np.linalg.norm(dx) - d <= epsilon:
            return 0
        discriminant = dx_dot_v ** 2 + d ** 2 - np.linalg.norm(dx) ** 2
        if discriminant > 0:
            dist: float = dx_dot_v - np.sqrt(discriminant)
            if dist <= total_step and dist >= 0: return dist
        return float('inf')
