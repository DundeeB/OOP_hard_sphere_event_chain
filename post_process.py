import random
import numpy as np
from SnapShot import *
from Structure import *
from EventChainActions import *
from sklearn.neighbors import *
import os


class OrderParameter:

    def __init__(self, sim_path):
        self.sim_path = sim_path
        write_or_load = WriteOrLoad(sim_path)
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = write_or_load.load_Input()
        self.event_2d_cells = Event2DCells(edge, n_row, n_col)
        self.event_2d_cells.add_third_dimension_for_sphere(l_z)
        centers, self.spheres_ind = write_or_load.last_spheres()
        self.event_2d_cells.append_sphere([Sphere(c, rad) for c in centers])
        self.write_or_load = WriteOrLoad(sim_path, self.event_2d_cells.boundaries)
        self.N = len(centers)
        self.op_vec = None

    def calc_order_parameter(self):
        """to be override by child class"""
        self.op_vec = [not None for _ in range(self.N)]
        self.op_name = "phi"
        pass

    def correlation(self, n_pairs):
        sp = self.event_2d_cells.all_spheres
        pairs = []
        pairs_dr = []
        counter = 0
        while len(pairs) < n_pairs:
            counter += 1
            i1, i2 = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            while i1 == i2: i1, i2 = random.randint(0, self.N - 1), random.randint(0, self.N - 1)
            j = min(i1, i2)
            k = max(i1, i2)
            if (j, k) not in pairs:
                pairs_dr.append(Metric.cyclic_dist(self.event_2d_cells.boundaries, sp[j], sp[k]))
                pairs.append((j, k))
            if counter > n_pairs ** 3:
                print("Too many cycles searching for pairs. Asker for " + str(n_pairs) + " pairs, after " + str(
                    n_pairs ** 3) + "cycles found only " + str(len(pairs)) + "pairs.")
                break
        if self.op_vec is None: self.calc_order_parameter()
        phiphi = []
        for (j, k) in pairs:
            phiphi.append(self.op_vec[j] * np.conj(self.op_vec[k]))
        return phiphi, pairs_dr, pairs

    def calc_write(self, n_pairs):
        if self.op_vec is None: self.calc_order_parameter()
        phiphi, pairs_dr, pairs = self.correlation(n_pairs)
        f = lambda a, b: os.path.join(a, b)
        np.savetxt(f(f(self.sim_path, "OP"), self.op_name + "_" + str(self.spheres_ind)), self.op_vec)
        np.savetxt(f(f(self.sim_path, "OP"), "pairs_dist_" + str(self.spheres_ind)), pairs_dr)
        np.savetxt(f(f(self.sim_path, "OP"), self.op_name + "_correlation_" + str(self.spheres_ind)), phiphi)


class PsiMN(OrderParameter):

    def __init__(self, sim_path, m, n):
        super().__init__(sim_path)
        self.m, self.n = m, n

    def calc_order_parameter(self):
        centers = self.event_2d_cells.all_centers
        sp = self.event_2d_cells.all_spheres
        cyc = lambda p1, p2: Metric.cyclic_dist(self.event_2d_cells.boundaries, p1, p2)
        self.graph = kneighbors_graph([p[:2] for p in centers], n_neighbors=self.n, metric=cyc)
        psimn_vec = np.zeros(self.N)
        for i in range(self.N):
            sp[i].nearest_neighbors = [sp[j] for j in self.graph.getrow(i).indices]
            dr = [np.array(centers[i]) - s.center for s in sp[i].nearest_neighbors]
            t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
            psi_n = np.mean(np.exp(1j * self.n * t))
            psimn_vec[i] = np.abs(psi_n) * np.exp(1j * self.m * np.angle(psi_n))
        self.op_vec = psimn_vec
        self.op_name = "psi_" + str(self.m) + str(self.n)
