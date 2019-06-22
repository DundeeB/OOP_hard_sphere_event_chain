import numpy as np
import random

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
        :return: True if they overlap
        """
        return np.linalg.norm(sphere1.center-sphere2.center) < sphere1.rad + sphere2.rad

    @staticmethod
    def spheres_overlap(spheres):
        for i in range(len(spheres)):
            for j in range(i):
                if Sphere.overlap(spheres[i],spheres[j]):
                    return True
        return False


class Cell:

    def __init__(self, site, edges, spheres=[]):
        """
        Cell object is a d-dimension cube in space, containing list of Sphere objects
        :param site: left bottom (in 3d back) corner of the cell
        :param edges: d dimension list of edges length for the cell
        :param spheres: list of d-dimension Sphere objects in the cell.
        """
        self.site = site
        self.edges = edges
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

    def random_generate_spheres(self,n_spheres, rads,extra_edges=[]):
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


class ArrayOfCells:

    def __init__(self, cells=[]):
        self.cells = cells

    @staticmethod
    def construct_default_2d_cells(n_rows, n_columns, l_x, l_y):
        """
        Construct a 2 dimension defualt choice list of empty cells (without spheres)
        :param n_rows: number of rows in the array of cells
        :param n_columns: number of colums in the array of cells
        :param l_x: physical length of the System in the x dimension
        :param l_y: physical length of the System in the y dimension
        :return: list of cells. cells[i][j] are in row i and column j
        """
        cells = [[[] for _ in range(n_columns)] for _ in range(n_rows)]
        edges = [l_x/n_columns, l_y/n_rows]
        for i in range(n_rows):
            for j in range(n_columns):
                site = [l_x/n_columns*i, l_y/n_rows*j]
                cells[i][j] = Cell(site, edges)
        return cells

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
    def overlap_2_cells(cell1,cell2):
        """
        Checks if the spheres in cell1 and cell2 are overlapping with each other. Does not check inside cell
        :param cell1: Cell object
        :param cell2: Cell object
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
