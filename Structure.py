import numpy as np
import random
from enum import Enum
from EventChainActions import *
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
        r = self.center+np.array(v_hat)*t
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
        ps = [self.center]
        t = 0
        sphere_to_move_around = Sphere(self.center, self.rad)
        while t < total_step:
            dist_to_wall = Metric.dist_to_boundary_without_r(sphere_to_move_around, v_hat, boundaries)
            if dist_to_wall < total_step - t:
                new_location = sphere_to_move_around.trajectory(dist_to_wall, v_hat, boundaries)
                ps.append(new_location)
                sphere_to_move_around.center = new_location
                t = t + dist_to_wall
            else:
                ps.append(self.trajectory(total_step, v_hat, boundaries))
                t = total_step
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

    def remove_sphere(self,spheres_to_remove):
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

    @staticmethod
    def construct_default_2d_cells(n_rows, n_columns, boundaries):
        """
        Construct a 2 dimension defualt choice list of empty cells (without spheres)
        :param n_rows: number of rows in the array of cells
        :param n_columns: number of colums in the array of cells
        :type boundaries: CubeBoundaries
        :return: list of cells. cells[i][j] are in row i and column j
        """
        l_x = boundaries.edges[0]
        l_y = boundaries.edges[1]
        cells = [[[] for _ in range(n_columns)] for _ in range(n_rows)]
        edges = [l_x/n_columns, l_y/n_rows]
        for i in range(n_rows):
            for j in range(n_columns):
                site = [l_x/n_columns*i, l_y/n_rows*j]
                cells[i][j] = Cell(site, edges, ind=(i, j))
        return ArrayOfCells(2, boundaries, cells)

    def closest_site_2d(self, point, edge):
        """
        Solve for closest site to point, in 2d, assuming all cells edges have the same length edge
        :return: tuple (i,j) of the closest cell = cells[i][j]
        """
        i = round(point[1] % edge) % len(self.cells)
        j = round(point[0] % edge) % len(self.cells[i])
        return i, j

    def relevant_cells_around_point_2d(self, rad, point, edge):
        """
        Finds cells which a sphere with radius rad with center at point would have overlap with
        :param rad: radius of interest
        :param point: point around which we find cells
        :param edge: assumes constant edge length of all cells
        :return: list of cells with overlap with
        """
        lx = self.boundaries.edges[0]
        ly = self.boundaries.edges[1]
        i, j = self.closest_site_2d(point, edge)
        a = self.cells[i][j]
        b = self.cells[i - 1][j]
        c = self.cells[i - 1][j - 1]
        d = self.cells[i][j - 1]
        dx, dy = np.array(point)-self.cells[i][j].site

        #Boundaries:
        if dx > edge: dx = dx - lx
        if dy > edge: dy = dy - ly
        if dx < -edge: dx = lx + dx
        if dy < -edge: dy = ly + dy

        # cases for overlap
        if dx > rad and dy > rad:  # 0<theta<90  # 1
            return [a]
        if dx > rad and abs(dy) <= rad:  # 0=theta  # 2
            return [a, b]
        if dx > rad and dy < -rad:  # -90<theta<0  # 3
            return [b]
        if abs(dx) <= rad and dy < -rad:  # theta=-90  # 4
            return [b, c]
        if dx < -rad and dy < -rad:  # -180<theta<-90  # 5
            return [c]
        if dx < -rad and abs(dy) <= rad:  # theta=180  # 6
            return [c, d]
        if dx < -rad and dy > rad:  # 90<theta<180  # 7
            return [d]
        if abs(dx) <= rad and dy > rad:  # theta=90  # 8
            return [a, d]
        else:  # abs(dx) <= rad and abs(dy) <= rad:  # x=y=0  # 9
            return [a, b, c, d]

    def get_all_crossed_points_2d(self, sphere, total_step, v_hat, edge):
        """

        :type sphere: Sphere
        :param total_step:
        :param v_hat:
        :param edge: assumes constant edge length of all cells
        :return:
        """
        vx = np.dot(v_hat, [1, 0])
        vy = np.dot(v_hat, [0, 1])
        ts = [0]
        l = sphere.systems_length_in_v_direction(v_hat, self.boundaries)
        if vy != 0:
            for i in range(len(self.cells)):
                y = i*edge
                t = (y - np.dot(sphere.center, [0, 1])) / vy
                if t < 0: t = t + l
                ts.append(t)
        if vx != 0:
            for j in range(len(self.cells[0])):
                x = j*edge
                t = (x - np.dot(sphere.center, [1, 0])) / vx
                if t < 0: t = t + l
                ts.append(t)
        return np.sort([t for t in ts if t < total_step])

    def perform_step(self, cell_ind, sphere, total_step, v_hat, edge):
        """
        Figures out the proper step for sphere inside cell which is at cells[cell_ind]
        :param cell_ind: (i,j,...)
        :type sphere: Sphere
        :param total_step: total step left to perform
        :param v_hat: direction of step
        :param edge: assumes constant edge length of all cells
        """
        cell = self.cell_from_ind(cell_ind)
        cell.remove(sphere)
        cells = []
        sub_cells = []
        for t in self.get_all_crossed_points_2d(sphere, total_step, v_hat, edge):
            previous_sub_cells = sub_cells
            sub_cells = []
            for c in self.relevant_cells_around_point_2d(sphere.rad, sphere.trajectory(t, v_hat, self.boundaries), edge):
                if c not in previous_sub_cells:
                    sub_cells.append(c)
            cells.append(sub_cells)
        event_chain_actions = EventChainActions(self.boundaries)
        for sub_cells in cells:
            other_spheres = [s for s in c.spheres for c in sub_cells]
            event = event_chain_actions.next_event(sphere, other_spheres, total_step)
            if event.event_type != EventType.FREE: break
        event.step.perform_step(event.step, self.boundaries)
        for new_cell in sub_cells:
            if new_cell.should_sphere_be_in_cell(sphere):
                new_cell.add_sphere(sphere)
                break
        if event.event_type == EventType.COLLISION:
            self.perform_step(new_cell.ind, event.other_sphere, event.step.total_step)
        if event.event_type == EventType.WALL:
            event.step.v_hat = CubeBoundaries.flip_v_hat_wall_part(event.wall,sphere,v_hat)
            self.perform_step(new_cell.ind, sphere, event.step.total_step)
        if event.event_type == EventType.FREE:
            return

