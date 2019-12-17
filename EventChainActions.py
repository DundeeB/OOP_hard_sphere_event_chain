from Structure import *
#from SnapShot import View2D


epsilon = 1e-10


class EventType(Enum):
    FREE = "FreeStep"  # Free path, end of step
    COLLISION = "SphereSphereCollision"  # Path leads to collision with another sphere
    WALL = "RigidWallBoundaryCondition"  # Path reaches rigid wall and needs to be handle
    PASS = "PassSphereBetweenCells"  # pass sphere between cells and let the new cell take it from there


class Event:

    def __init__(self, event_type: EventType, other_sphere: Sphere, wall):
        """
        A single event that will happen to sphere, such as collision with another sphere
        or arriving at the boundary of the simulation
        :type event_type: EventType
        :param other_sphere: if event is collision, this is the sphere it will collide
        :type other_sphere: Sphere
        :param wall: if event is boundary condition, this is the wall it is going to reach
        """
        self.event_type = event_type
        self.other_sphere = other_sphere
        self.wall = wall


class Step:

    def __init__(self, sphere: Sphere, total_step, v_hat, boundaries: CubeBoundaries, current_step=np.nan):
        """
        The characteristic of the step to be perform
        :type sphere: Sphere
        :param total_step: total step left for the current move of spheres
        :param current_step: step sphere is about to perform
        :param v_hat: direction of the step
        :type boundaries: CubeBoundaries
        """
        self.sphere = sphere
        self.total_step = total_step
        self.current_step = current_step
        self.v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
        self.boundaries = boundaries

    def perform_step(self):
        """
        Perform the current step (calls Sphere's perform step), and subtract step from total step
        """
        if np.isnan(self.current_step):
            raise ValueError("Current step is nan and step is about to occur")
        assert self.current_step <= self.total_step
        self.sphere.perform_step(self.v_hat, self.current_step, self.boundaries)
        self.total_step = self.total_step - self.current_step

    def next_event(self, other_spheres):
        """
        Returns the next Event object to be handle, such as perform the step and decide the next event
        :param other_spheres: other spheres which sphere might collide
        :return: Event object containing the information about the event about to happen after the step, such as step
        size or step type (wall free or boundary), and the current step
        """
        sphere, total_step, v_hat, current_step = self.sphere, self.total_step, self.v_hat, self.current_step
        min_dist_to_wall, closest_wall = Metric.dist_to_boundary(sphere, total_step, v_hat, self.boundaries)
        closest_sphere = []
        closest_sphere_dist = float('inf')
        for other_sphere in other_spheres:
            if other_sphere == sphere:
                continue
            sphere_dist = Metric.dist_to_collision(sphere, other_sphere, total_step, v_hat, self.boundaries)
            if sphere_dist < closest_sphere_dist:
                closest_sphere_dist = sphere_dist
                closest_sphere = other_sphere
        # it hits a wall
        if min_dist_to_wall < closest_sphere_dist:
            if np.isnan(current_step) or min_dist_to_wall < current_step:
                self.current_step = min_dist_to_wall
            return Event(EventType.WALL, [], closest_wall), min_dist_to_wall
        # it hits another sphere
        if min_dist_to_wall > closest_sphere_dist:
            if np.isnan(current_step) or closest_sphere_dist < current_step:
                self.current_step = closest_sphere_dist
            return Event(EventType.COLLISION, closest_sphere, []), closest_sphere_dist
        # it hits nothing, both min_dist_to_wall and closest_sphere_dist are inf
        if np.isnan(current_step) or total_step < current_step:
            self.current_step = total_step
            return Event(EventType.FREE, [], []), total_step
        else:  # total_step > current_step
            return Event(EventType.PASS, [], []), current_step


class Event2DCells(ArrayOfCells):

    def __init__(self, edge, n_rows, n_columns):
        """
        Construct a 2 dimension default choice list of empty cells (without spheres), with constant edge
        :param n_rows: number of rows in the array of cells
        :param n_columns: number of columns in the array of cells
        :param edge: constant edge size of all cells is assumed and needs to be declared
        """
        l_x = edge * n_columns
        l_y = edge * n_rows
        cells = [[[] for _ in range(n_columns)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_columns):
                site = (edge*j, edge*i)
                cells[i][j] = Cell(site, [edge, edge], ind=(i, j))
                cells[i][j].spheres = []
        boundaries = CubeBoundaries([l_x, l_y], [BoundaryType.CYCLIC, BoundaryType.CYCLIC])
        super().__init__(2, boundaries, cells=cells)
        self.edge = edge
        self.l_x = l_x
        self.l_y = l_y
        self.l_z = None

    def add_third_dimension_for_sphere(self, l_z):
        self.l_z = l_z
        self.boundaries = CubeBoundaries(self.boundaries.edges + [l_z], \
                                         self.boundaries.boundaries_type + [BoundaryType.WALL])
        return

    def random_generate_spheres(self, n_spheres_per_cell, rad, extra_edges=[]):
        if extra_edges == [] and self.l_z is not None:
            super().random_generate_spheres(n_spheres_per_cell, rad, extra_edges=[self.l_z])
        else:
            super().random_generate_spheres(n_spheres_per_cell, rad, extra_edges)

    def generate_spheres_in_cubic_structure(self, n_spheres_per_cell, rad, extra_edges=[]):
        if extra_edges == [] and self.l_z is not None:
            super().generate_spheres_in_cubic_structure(n_spheres_per_cell, rad, extra_edges=[self.l_z])
        else:
            super().generate_spheres_in_cubic_structure(n_spheres_per_cell, rad, extra_edges)

    def closest_site_2d(self, point):
        """
        Solve for closest site to point,
        :return: tuple (i,j) of the closest cell = cells[i][j]
        """
        i = int(round(point[1] / self.edge) % self.n_rows)
        j = int(round(point[0] / self.edge) % self.n_columns)
        return i, j

    def maximal_free_step(self, i, j, step: Step):
        """
        Returns the maximal free step allowed so the sphere would pass between the cells, without overlapping outside
        the new cell
        :type step: Step
        :return: the corresponding maximal free step allowed
        """
        p, v_hat, e, c, r = step.sphere.center, step.v_hat, self.edge, self.cells[i][j].site, step.sphere.rad
        xp, xm, yp, ym = c[0] + 2*e, c[0] - e, c[1] + 2*e, c[1] - e

        if v_hat[0] >= 0: x = xp - r
        else: x = xm
        if v_hat[1] >= 0: y = yp - r
        else: y = ym

        if v_hat[0] != 0:
            tx = (float) (x - p[0])/v_hat[0]
        else: tx = float('inf')

        if v_hat[1] != 0:
            ty = (float) (y - p[1])/v_hat[1]
        else: ty = float('inf')

        t = min(tx, ty)
        return t

    def perform_total_step(self, i, j, step: Step, draw=None):
        """
        Perform step for all the spheres, starting from sphere inside cell
        :type step: Step
        :type draw: View2D
        """
        if draw is not None:
            if draw.counter is not None:
                draw.counter += 1
                img_name = str(draw.counter)
            else:
                img_name = 'total_step=' + str(round(step.total_step, 4))
            draw.array_of_cells_snapshot('During step snapshot, total_step=' + str(step.total_step),
                                         self, img_name, step)

        sphere, total_step, v_hat, cell = step.sphere, step.total_step, step.v_hat, self.cells[i][j]
        step.current_step = np.nan
        v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
        cell.remove_sphere(sphere)

        other_spheres = []
        relevant_cells = [self.cells[i][j]] + self.neighbors(i, j)
        for c in relevant_cells:
            for s in c.spheres: other_spheres.append(s)
        step.current_step = self.maximal_free_step(i, j, step)
        event, current_step = step.next_event(other_spheres)

        assert event is not None and not np.isnan(step.current_step)

        step.perform_step()  # subtract current step from total step

        new_cell, flag = None, None
        for new_cell in relevant_cells:
            if new_cell.center_in_cell(sphere):
                new_cell.append(sphere)
                flag = not None
                break
        assert flag is not None, "sphere has not been added to any cell"
        i_n, j_n = new_cell.index[0],new_cell.index[1]

        if event.event_type == EventType.COLLISION:
            new_cell, flag = None, None
            for new_cell in relevant_cells:
                if new_cell.center_in_cell(event.other_sphere):
                    flag = not None
                    break
            assert flag is not None, "Did not find new cell for the collided sphere"
            step.sphere = event.other_sphere
            i_n, j_n = new_cell.index[0], new_cell.index[1]
            self.perform_total_step(i_n, j_n, step, draw)
            return
        if event.event_type == EventType.WALL:
            step.v_hat = CubeBoundaries.flip_v_hat_at_wall(event.wall, sphere, v_hat)
            self.perform_total_step(i_n, j_n, step, draw)
            return
        if event.event_type == EventType.PASS:
            self.perform_total_step(i_n, j_n, step, draw)
        if event.event_type == EventType.FREE: return
