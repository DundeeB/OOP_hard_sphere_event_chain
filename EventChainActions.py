from Structure import *

# import time

epsilon = 1e-8


class EventType(Enum):
    FREE = "FreeStep"  # Free path, end of step
    COLLISION = "SphereSphereCollision"  # Path leads to collision with another sphere
    WALL = "RigidWallBoundaryCondition"  # Path reaches rigid wall and needs to be handle
    PASS = "PassSphereBetweenCells"  # pass sphere between cells and let the new cell take it from there


class Event:

    def __init__(self, event_type: EventType, other_sphere: Sphere):
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


class Step:

    def __init__(self, sphere: Sphere, total_step, direction: Direction, boundaries,
                 current_step=np.nan):
        """
        The characteristic of the step to be perform
        :type sphere: Sphere
        :param total_step: total step left for the current move of spheres
        :param current_step: step sphere is about to perform
        :param direction: of the step
        :type boundaries: list
        """
        self.sphere = sphere
        self.total_step = total_step
        self.current_step = current_step
        self.direction = direction
        self.boundaries = boundaries

    def next_event(self, other_spheres, cut_off=float('inf')):
        """
        Returns the next Event object to be handle, such as perform the step and decide the next event
        :param other_spheres: other spheres which sphere might collide
        :return: Event object containing the information about the event about to happen after the step, such as step
        size or step type (wall free or boundary), and the current step
        """
        sphere, total_step, direction = self.sphere, self.total_step, self.direction
        min_dist_to_wall = Metric.dist_to_wall(sphere, total_step, direction, self.boundaries)
        closest_sphere_dist, closest_sphere = Metric.dist_to_collision(sphere, other_spheres, total_step, direction,
                                                                       self.boundaries, cut_off)
        case = np.argmin([min_dist_to_wall, closest_sphere_dist, total_step, self.current_step])
        if case == 0:  # it hits a wall
            self.current_step = min_dist_to_wall
            return Event(EventType.WALL, [])
        if case == 1:  # it hits another sphere
            self.current_step = closest_sphere_dist
            return Event(EventType.COLLISION, closest_sphere)
        if case == 2:  # it hits nothing, both min_dist_to_wall and closest_sphere_dist are inf
            self.current_step = total_step
            return Event(EventType.FREE, [])
        else:  # total_step > current_step
            return Event(EventType.PASS, [])  # do not update step.current_step, because next step is PASS and he
            # would not actually perform total step


class Event2DCells(ArrayOfCells):

    def __init__(self, edge, n_rows, n_columns, l_z):
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
                site = (edge * j, edge * i)
                cells[i][j] = Cell(site, [edge, edge], ind=(i, j))
                cells[i][j].spheres = []
        boundaries = [l_x, l_y, l_z]
        super().__init__(2, boundaries, cells=cells)
        self.edge = edge
        self.l_x = l_x
        self.l_y = l_y
        self.l_z = l_z

    def cell_of_sphere(self, sphere):
        return self.cells[int(np.floor(sphere.center[1] / self.edge))][int(np.floor(sphere.center[0] / self.edge))]

    def append_sphere(self, spheres):
        if type(spheres) != list:
            assert type(spheres) == Sphere
            spheres = [spheres]
        cells = []
        for sphere in spheres:
            cell = self.cell_of_sphere(sphere)
            cell.append(sphere)
            cells.append(cell)
        if len(cells) == 1:
            return cells[0]
        return cells

    def maximal_free_step(self, i, j, step: Step):
        """
        Returns the maximal free step allowed so the sphere would pass between the cells, without overlapping outside
        the new cell
        :type step: Step
        :return: the corresponding maximal free step allowed
        """
        if step.direction.dim == 2:
            return float('inf')
        else:
            return float(
                self.cells[i][j].site[step.direction.dim] + 2 * self.edge - 2 * step.sphere.rad - step.sphere.center[
                    step.direction.dim])

    def perform_total_step(self, i, j, step: Step, draw=None, record_displacements=False):
        """
        Perform step for all the spheres, starting from sphere inside cell
        :param i: indices of the cell containing the sphere trying to make a move
        :param j: indices of the cell containing the sphere trying to make a move
        :type step: Step
        :type draw: WriteOrLoad
        """
        if record_displacements:
            displacements = 0
            # init_time = time.time()  # use record displacement as debug mode switch

        while step.total_step > 0:
            # if time.time() - init_time > 10:
            #     print("A single step is taking more than 10s")
            if draw is not None:
                if draw.counter is not None:
                    draw.counter += 1
                    img_name = str(draw.counter)
                else:
                    img_name = 'total_step=' + str(round(step.total_step, 4))
                draw.array_of_cells_snapshot('During step snapshot, total_step=' + str(step.total_step),
                                             self, img_name, step)
                draw.dump_spheres(self.all_centers, img_name)

            sphere, direction, cell = step.sphere, step.direction, self.cells[i][j]
            cell.remove_sphere(sphere)

            other_spheres = [s for s in cell.spheres]

            def add(c):
                for s in c.spheres:
                    other_spheres.append(s)

            ip1, jp1, im1, jm1 = ArrayOfCells.cyclic_indices(i, j, self.n_rows, self.n_columns)
            if direction.dim == 0:
                for c in [self.cells[ip1][j], self.cells[ip1][jp1], self.cells[i][jp1],
                          self.cells[im1][jp1], self.cells[im1][j]]:
                    add(c)
            else:
                if direction.dim == 1:
                    for c in [self.cells[i][jm1], self.cells[ip1][jm1], self.cells[ip1][j],
                              self.cells[ip1][jp1], self.cells[i][jp1]]:
                        add(c)
                else:
                    for c in [self.cells[ip1][jm1], self.cells[ip1][j], self.cells[ip1][jp1], self.cells[i][jp1],
                              self.cells[i][jm1], self.cells[im1][jm1], self.cells[im1][j], self.cells[im1][jp1]]:
                        add(c)
            step.current_step = self.maximal_free_step(i, j, step)
            event = step.next_event(other_spheres, cut_off=self.edge)  # updates step.current_step
            step.sphere.perform_step(direction, step.current_step, self.boundaries)
            step.total_step = step.total_step - step.current_step
            if record_displacements:
                displacements += 1
            new_cell = self.append_sphere(sphere)
            i, j = new_cell.ind[:2]
            if event.event_type == EventType.COLLISION:
                step.sphere = event.other_sphere
                i, j = self.cell_of_sphere(step.sphere).ind[:2]
                continue
            if event.event_type == EventType.WALL:
                step.direction.sgn = -1 * step.direction.sgn
                continue
            if event.event_type == EventType.PASS:
                continue
            if event.event_type == EventType.FREE:
                if record_displacements: return displacements
                return

    def perform_LMC_step(self, i_cell, j_cell, sphere, step_vector):
        # maximal_free_step = min(
        #     [self.cells[i][j].site[dim] + 2 * self.edge - 2 * sphere.rad - sphere.center[dim] for dim in range(2)])
        # if np.linalg.norm(step_vector) > maximal_free_step:
        #     print('problem with maximal free step')

        cell = self.cells[i_cell][j_cell]
        cell.remove_sphere(sphere)
        old_center = copy.deepcopy(sphere.center)
        for i in range(len(sphere.center)):
            sphere.center[i] += step_vector[i]
        if sphere.center[2] > self.boundaries[2] - sphere.rad or sphere.center[2] < sphere.rad:
            sphere.center = old_center
            self.append_sphere(sphere)
            return False
        # box it only for dim=0,1
        for i in range(2):
            sphere.center[i] = sphere.center[i] % self.boundaries[i]
        i_cell, j_cell = int(np.floor(sphere.center[1] / self.edge)), int(np.floor(sphere.center[0] / self.edge))
        new_cell = self.cells[i_cell][j_cell]
        ip1, jp1, im1, jm1 = ArrayOfCells.cyclic_indices(i_cell, j_cell, self.n_rows, self.n_columns)
        for c in [new_cell, self.cells[ip1][jm1], self.cells[ip1][j_cell], self.cells[ip1][jp1],
                  self.cells[i_cell][jp1], self.cells[i_cell][jm1], self.cells[im1][jm1], self.cells[im1][j_cell],
                  self.cells[im1][jp1]]:
            for other_sphere in c.spheres:
                if Metric.overlap(sphere, other_sphere, self.boundaries):
                    sphere.center = old_center
                    cell.append(sphere)
                    return False
        new_cell.append(sphere)
        return True

    def generate_spheres_in_AF_triangular_structure(self, n_row, n_col, rad):
        """
        For 3D case, created spheres in the 6-fold comb lattice Anti-Ferromagnetic ground state
        :param n_row: create 2 n_row/2 triangular lattices
        :param n_col: same, each triangular lattice has n_col columns
        :param rad: not a list, a single number of the same radius for all spheres
        """
        assert type(rad) != list, "list of different rads is not supported for initial condition AF triangular"
        l_x, l_y, l_z = self.boundaries
        assert n_row % 2 == 0, "n_row should be even for anti-ferromagnetic triangular Initial conditions"
        ay = 2 * l_y / n_row
        spheres_down = ArrayOfCells.spheres_in_triangular(int(n_row / 2), n_col, rad, l_x, l_y)
        spheres_up = ArrayOfCells.spheres_in_triangular(int(n_row / 2), n_col, rad, l_x, l_y)
        dz_max = np.sqrt(max((2 * rad) ** 2 - (2.0 / 3.0 * ay) ** 2, 0))
        dr_max = (self.l_z / 2 - rad - dz_max / 2) / 2  # by two to not have coincidence collision
        for s in spheres_down:
            assert type(s) == Sphere
            cx, cy = s.center[:2]
            s.center = [cx, cy, rad + dr_max * np.random.random()]
        for s in spheres_up:
            assert type(s) == Sphere
            cx, cy = s.center[:2]
            s.center = [cx, cy + ay * 2.0 / 3, l_z - rad - dr_max * np.random.random()]
            s.box_it(self.boundaries)
        self.append_sphere(spheres_down + spheres_up)
        self.update_all_spheres()
        assert self.legal_configuration()

    def generate_spheres_in_AF_square(self, n_sp_row, n_sp_col, rad):
        """
        For 3D case, created spheres in the 6-fold comb lattice Anti-Ferromagnetic ground state
        :param n_spheres_per_cell: number of total sphere in each cell
        :param rad: not a list, a single number of the same radius for all spheres
        """
        assert type(rad) != list, "list of different rads is not supported for initial condition AF triangular"
        sig = 2 * rad
        ax, ay = self.l_x / n_sp_col, self.l_y / n_sp_row
        a = min(ax, ay)
        assert a ** 2 + (self.l_z - sig) ** 2 > sig ** 2 and 4 * a ** 2 > sig ** 2, \
            "Can not create so many spheres in the AF square lattice"
        spheres = []
        dz_max = np.sqrt(max(sig ** 2 - a ** 2, 0))
        dr_max = self.l_z / 2 - rad - dz_max / 2  # by two to not have coincidence collision
        for i in range(n_sp_row):
            for j in range(n_sp_col):
                sign = (-1) ** (i + j)
                r = rad + 100 * epsilon
                dr = np.random.random() * dr_max
                x, y, z = (j + 1 / 2) * ax, (i + 1 / 2) * ay, sign * (r + dr) + self.l_z * (1 - sign) / 2
                spheres.append(Sphere([x, y, z], rad))
        self.append_sphere(spheres)
        self.update_all_spheres()
        assert self.legal_configuration()

    def scale_xy(self, factor):
        """
        scale xy dimensions by multiplying by factor.
        Does not scale the radius of the spheres and the z dimension
        :param factor: factor>1 means bigger simulation and cells
        :return:
        """
        self.edge *= factor
        self.l_x *= factor
        self.l_y *= factor
        # not self.l_z
        self.boundaries[0] *= factor
        self.boundaries[1] *= factor
        # not self.boundaries[2]
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                c = self.cells[i][j]
                c.site = [x * factor for x in c.site]
                c.edges = [e * factor for e in c.edges]  # cell is 2D
                for s in c.spheres:
                    s.center = [s.center[0] * factor, s.center[1] * factor, s.center[2]]
        assert self.legal_configuration(), "Scaling failed, illegal configuration"

    def quench(self, desired_rho):
        """
        Moving spheres far from the boundary conditions andclosing the boundary conditions until the
        density becomes dest_rho. If dest_rho is larger then current rho we simply scale.
        Quite similiar to Simulated Annealing, and the method name could be annealing as well. As formally I simply
        rapidly change the boundaries in order to get the desiered density, with no equilibration in the proccess,
        simulated Annealing is a bit more specific then that, I chose the name quench.
        :param desired_rho: density destination
        """
        spheres = self.all_spheres
        N = len(spheres)
        rho = N * ((2 * spheres[0].rad) ** 3) / (self.l_x * self.l_y * self.l_z)
        while rho < desired_rho:  # we need to compress
            min_x, i_min_x = min((s.center[0] - s.rad, i_sp) for (i_sp, s) in enumerate(spheres))
            max_x, i_max_x = max((s.center[0] + s.rad, i_sp) for (i_sp, s) in enumerate(spheres))
            min_y, i_min_y = min((s.center[1] - s.rad, i_sp) for (i_sp, s) in enumerate(spheres))
            max_y, i_max_y = max((s.center[1] + s.rad, i_sp) for (i_sp, s) in enumerate(spheres))
            new_lx = max_x - min_x
            new_ly = max_y - min_y
            if new_lx < self.l_x or new_ly < self.l_y:  # we have some space to squizz
                vec_to_zero = np.array([0.0, 0.0])
                if new_lx < self.l_x:
                    vec_to_zero[0] = -min_x
                    self.l_x = new_lx
                if new_ly < self.l_y:
                    vec_to_zero[1] = -min_y
                    self.l_y = new_ly
                all_spheres = self.translate(vec_to_zero)  # emptied all spheres from self
                self.boundaries = [self.l_x, self.l_y, self.l_z]
                self.n_rows, self.n_columns = int(np.ceil(self.l_y / self.edge)), int(np.ceil(self.l_x / self.edge))
                for i_sp in range(len(all_spheres)):
                    all_spheres[i_sp].box_it(self.boundaries)
                self.append_sphere(all_spheres)
                assert self.legal_configuration()
            else:  # we must move spheres around before squizzing
                indices = [i_min_y, i_min_x, i_max_y, i_max_x]
                for i_sp, direction in zip(indices, Direction.directions()):
                    sphere = self.all_spheres[i_sp]
                    # total_step = 10 * sphere.rad
                    total_step = sphere.rad / 2
                    step = Step(sphere, total_step, direction, self.boundaries)
                    cell = self.append_sphere(sphere)
                    i, j = cell.ind[:2]
                    self.perform_total_step(i, j, step)
            rho = N * ((2 * self.all_spheres[0].rad) ** 3) / (self.l_x * self.l_y * self.l_z)
        if rho > desired_rho:
            factor = np.sqrt(rho / desired_rho)  # >= 1
            self.scale_xy(factor)

    def z_quench(self, desired_lz):
        assert_exit = False
        while self.l_z > desired_lz:
            min_z, i_min_z = min((s.center[2] - s.rad, i_sp) for (i_sp, s) in enumerate(self.all_spheres))
            for s in self.all_spheres:
                s.center[2] -= min_z
            max_z, i_max_z = max((s.center[2] + s.rad, i_sp) for (i_sp, s) in enumerate(self.all_spheres))
            if max_z < desired_lz:
                max_z = desired_lz
                assert_exit = True
            self.l_z = max_z
            self.boundaries = [self.l_x, self.l_y, self.l_z]
            if assert_exit: return
            sp_down, sp_up = self.all_spheres[i_min_z], self.all_spheres[i_max_z]
            total_step = sp_up.rad / 2
            for direction in Direction.directions()[0:2]:
                step_sp_down = Step(sp_down, total_step, direction, self.boundaries)
                step_sp_up = Step(sp_up, total_step, direction if direction.dim != 2 else Direction.directions()[-1],
                                  self.boundaries)
                cell_down, cell_up = self.append_sphere(sp_down), self.append_sphere(sp_up)
                id, jd = cell_down.ind[:2]
                self.perform_total_step(id, jd, step_sp_down)
                iu, ju = cell_up.ind[:2]
                self.perform_total_step(iu, ju, step_sp_up)
