from Structure import *
#from SnapShot import View2D


epsilon = 1e-10


class EventType(Enum):
    FREE = "FreeStep"  # Free path
    COLLISION = "SphereSphereCollision"  # Path leads to collision with another sphere
    WALL = "RigidWallBoundaryCondition"  # Path reaches rigid wall and needs to be handle


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
        self.sphere.perform_step(self.v_hat, self.current_step, self.boundaries)
        self.total_step = self.total_step - self.current_step

    def next_event(self, other_spheres):
        """
        Returns the next Event object to be handle, such as from the even get the step, perform the step and decide the
        next event
        :param other_spheres: other spheres which sphere might collide
        :return: Event object containing the information about the event about to happen after the step, such as step
        size or step type (wall free or boundary), and the current step
        """
        sphere = self.sphere
        total_step = self.total_step
        v_hat = self.v_hat
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
            if np.isnan(self.current_step) or min_dist_to_wall < self.current_step:
                self.current_step = min_dist_to_wall
            return Event(EventType.WALL, [], closest_wall), min_dist_to_wall
        # it hits another sphere
        if min_dist_to_wall > closest_sphere_dist:
            if np.isnan(self.current_step) or closest_sphere_dist < self.current_step:
                self.current_step = closest_sphere_dist
            return Event(EventType.COLLISION, closest_sphere, []), closest_sphere_dist
        # it hits nothing, both min_dist_to_wall and closest_sphere_dist are inf
        if np.isnan(self.current_step) or total_step < self.current_step:
            self.current_step = total_step
        return Event(EventType.FREE, [], []), total_step


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
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.l_x = l_x
        self.l_y = l_y
        self.l_z = None

    def add_third_dimension_for_sphere(self, l_z):
        self.l_z = l_z
        self.boundaries = CubeBoundaries(self.boundaries.edges + [self.l_z], \
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
        Solve for closest site to point, in 2d, assuming all cells edges have the same length edge
        :return: tuple (i,j) of the closest cell = cells[i][j]
        """
        i = int(round(point[1] / self.edge) % self.n_rows)
        j = int(round(point[0] / self.edge) % self.n_columns)
        return i, j

    def cells_around_intersect_2d(self, point):
        """
        Finds cells which a sphere with radius rad with center at point would have direct_overlap with
        :param rad: radius of interest
        :param point: point around which we find cells
        :return: list of cells with direct_overlap with sphere
        """
        point = np.array([point[0], point[1]])
        point = point + np.array((epsilon, epsilon))
        x, y = point[0], point[1]
        e = self.edge
        i, j = int(y/e) % self.n_rows, int(x/e) % self.n_columns
        ip1, jp1, im1, jm1 = ArrayOfCells.cyclic_indices(i, j, self.n_rows, self.n_columns)
        return [self.cells[a][b] for a in [im1, i, ip1] for b in [jm1, j, jp1]]

    def get_all_crossed_points_2d(self, step: Step):
        """
        :param step: Step structure containing the information such as which sphere, v_hat and total step
        :return: list of ts such that trajectory(t) is a point crossing cell boundary
        """
        sphere = copy.deepcopy(step.sphere)
        total_step, v_hat = step.total_step, step.v_hat
        vx = v_hat[0]
        vy = v_hat[1]
        ts = [0]
        starting_points, starting_ts = sphere.trajectories_braked_to_lines(total_step, v_hat, self.boundaries)
        for starting_point, starting_t in zip(starting_points, starting_ts):
            if vy != 0:
                for i in range(self.n_rows + 1):  # +1 for last row/col
                    y = float(i * self.edge)
                    t = (y - starting_point[1]) / vy
                    current_t = starting_t + t
                    if t < 0 or current_t > total_step:
                        continue
                    ts.append(current_t)
            if vx != 0:
                for j in range(self.n_columns + 1):
                    x = float(j*self.edge)
                    t = (x - starting_point[0]) / vx
                    current_t = starting_t + t
                    if t < 0 or current_t > total_step:
                        continue
                    ts.append(current_t)
        sorted_ts = np.sort(list(dict.fromkeys([t for t in ts if t <= total_step])))
        if sorted_ts[-1] != total_step:
            sorted_ts = [t for t in sorted_ts] + [total_step]
        return sorted_ts

    def perform_total_step(self, cell: Cell, step: Step, draw=None):
        """
        Perform step for all the spheres, starting from sphere inside cell
        :type cell: Cell
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

        sphere, total_step, v_hat = step.sphere, step.total_step, step.v_hat
        step.current_step = np.nan
        v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
        cell.remove_sphere(sphere)
        cells, sub_cells, all_cells_on_traject = [], [], []  # list of sub_cells, sub_cells is a list of cells
        ts = self.get_all_crossed_points_2d(step)
        for t in ts:
            sub_cells = []
            for c in self.cells_around_intersect_2d(sphere.trajectory(t, v_hat, self.boundaries)):
                if c not in all_cells_on_traject:
                    sub_cells.append(c)
                    all_cells_on_traject.append(c)
            if sub_cells != []: cells.append(sub_cells)

        final_event, relevent_sub_cells = None, None
        minimal_step = float('inf')
        for i, sub_cells in enumerate(cells):
            other_spheres = []
            for c in sub_cells:
                for s in c.spheres: other_spheres.append(s)
            event, current_step = step.next_event(other_spheres)
            if current_step < minimal_step:
                final_event = event
                minimal_step = current_step
                relevent_sub_cells = sub_cells
            if event.event_type != EventType.FREE and i != len(cells) - 1:
                next_other_spheres = []
                for next_cell in cells[i+1]:
                    for s in next_cell.spheres: next_other_spheres.append(s)
                another_potential_event, another_current_step = step.next_event(next_other_spheres)
                if another_current_step < minimal_step:
                    final_event = another_potential_event
                    minimal_step = another_current_step
                    relevent_sub_cells = cells[i+1]
            if i == len(cells) - 1 or minimal_step < ts[i+1]:
                break
        sub_cells, event = relevent_sub_cells, final_event
        assert event is not None and not np.isnan(step.current_step)

        step.perform_step()  # subtract current step from total step

        new_cell, flag = None, None
        for new_cell in all_cells_on_traject:
            if new_cell.sphere_in_cell(sphere):
                new_cell.append(sphere)
                flag = not None
                break
        assert flag is not None, "sphere has not been added to any cell"

        if event.event_type == EventType.COLLISION:
            new_cell, flag = None, None
            for new_cell in all_cells_on_traject:
                if new_cell.sphere_in_cell(event.other_sphere):
                    flag = not None
                    break
            assert flag is not None, "Did not find new cell for the collided sphere"
            step.sphere = event.other_sphere
            self.perform_total_step(new_cell, step, draw)
            return
        if event.event_type == EventType.WALL:
            step.v_hat = CubeBoundaries.flip_v_hat_wall_part(event.wall, sphere, v_hat)
            self.perform_total_step(new_cell, step, draw)
            return
        if event.event_type == EventType.FREE: return
