import random
import numpy as np
from SnapShot import *
from Structure import *
from EventChainActions import *
from sklearn.neighbors import *
import os


class OrderParameter:

    def __init__(self, sim_path, centers=None, spheres_ind=None):
        self.sim_path = sim_path
        write_or_load = WriteOrLoad(sim_path)
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = write_or_load.load_Input()
        self.event_2d_cells = Event2DCells(edge, n_row, n_col)
        self.event_2d_cells.add_third_dimension_for_sphere(l_z)
        if centers is None or spheres_ind is None:
            centers, spheres_ind = write_or_load.last_spheres()
        self.spheres_ind = spheres_ind
        self.event_2d_cells.append_sphere([Sphere(c, rad) for c in centers])
        self.write_or_load = WriteOrLoad(sim_path, self.event_2d_cells.boundaries)
        self.N = len(centers)
        self.op_vec = None

    def calc_order_parameter(self):
        """to be override by child class"""
        self.op_vec = [not None for _ in range(self.N)]
        self.op_name = "phi"
        pass

    def correlation(self, bin_width=0.1):
        if self.op_vec is None: self.calc_order_parameter()

        phiphi_vec = (np.conj(np.transpose(np.matrix(self.op_vec))) *
                      np.matrix(self.op_vec)).reshape((len(self.op_vec) ** 2,))
        x = np.array([r[0] for r in self.event_2d_cells.all_centers])
        y = np.array([r[1] for r in self.event_2d_cells.all_centers])
        dx = (x.reshape((len(x), 1)) - x.reshape((1, len(x)))).reshape(len(x) ** 2, )
        dy = (y.reshape((len(y), 1)) - y.reshape((1, len(y)))).reshape(len(y) ** 2, )
        lx, ly = self.event_2d_cells.boundaries.edges[:2]
        dx = np.minimum(np.abs(dx), np.minimum(np.abs(dx + lx), np.abs(dx - lx)))
        dy = np.minimum(np.abs(dy), np.minimum(np.abs(dy + ly), np.abs(dy - ly)))
        pairs_dr = np.sqrt(dx ** 2 + dy ** 2)

        I = np.argsort(pairs_dr)
        pairs_dr = pairs_dr[I]
        phiphi_vec = phiphi_vec[0, I]

        centers = np.linspace(0, np.max(pairs_dr), int(np.max(pairs_dr) / bin_width) + 1) + bin_width / 2
        counts = np.zeros(len(centers))
        phiphi_hist = np.zeros(len(centers), dtype=np.complex)
        i = 0
        for j in range(len(pairs_dr)):
            if pairs_dr[j] > centers[i] + bin_width / 2:
                i += 1
            phiphi_hist[i] += phiphi_vec[0, j]
            counts[i] += 1
        I = np.where(np.logical_and(counts != 0, phiphi_hist != np.nan))
        counts = counts[I]
        phiphi_hist = np.real(phiphi_hist[I]) / counts + 1j * np.imag(phiphi_hist[I]) / counts
        centers = centers[I]
        return phiphi_hist, centers, counts, phiphi_vec, pairs_dr

    def calc_write(self, calc_correlation=True, bin_width=0.1):
        if self.op_vec is None: self.calc_order_parameter()
        f = lambda a, b: os.path.join(a, b)
        if not os.path.exists(f(self.sim_path, "OP")): os.mkdir(f(self.sim_path, "OP"))
        g = lambda name, mat: np.savetxt(
            f(f(self.sim_path, "OP"), self.op_name + "_" + name + "_" + str(self.spheres_ind)) + ".txt", mat)
        g("vec", self.op_vec)
        if calc_correlation:
            phi_hist, centers, counts, _, _ = self.correlation(bin_width=bin_width)
            g("correlation", np.transpose([centers, np.abs(phi_hist), counts]))


class PsiMN(OrderParameter):

    def __init__(self, sim_path, m, n):
        super().__init__(sim_path)
        self.m, self.n = m, n
        upper_centers = [c for c in self.event_2d_cells.all_centers if
                         c[2] >= self.event_2d_cells.boundaries.edges[2] / 2]
        lower_centers = [c for c in self.event_2d_cells.all_centers if
                         c[2] < self.event_2d_cells.boundaries.edges[2] / 2]
        self.lower = OrderParameter(sim_path, lower_centers, self.spheres_ind)
        self.upper = OrderParameter(sim_path, upper_centers, self.spheres_ind)

    @staticmethod
    def psi_m_n(event_2d_cells, m, n):
        centers = event_2d_cells.all_centers
        sp = event_2d_cells.all_spheres
        cyc_bound = CubeBoundaries(event_2d_cells.boundaries.edges[:2], 2 * [BoundaryType.CYCLIC])
        cyc = lambda p1, p2: Metric.cyclic_dist(cyc_bound, Sphere(p1, 1), Sphere(p2, 1))
        graph = kneighbors_graph([p[:2] for p in centers], n_neighbors=n, metric=cyc)
        psimn_vec = np.zeros(len(centers), dtype=np.complex)
        for i in range(len(centers)):
            sp[i].nearest_neighbors = [sp[j] for j in graph.getrow(i).indices]
            dr = [np.array(centers[i]) - s.center for s in sp[i].nearest_neighbors]
            t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
            psi_n = np.mean(np.exp(1j * n * t))
            psimn_vec[i] = np.abs(psi_n) * np.exp(1j * m * np.angle(psi_n))
        return psimn_vec, graph

    def calc_order_parameter(self):
        self.op_vec, _ = PsiMN.psi_m_n(self.event_2d_cells, self.m, self.n)
        self.lower.op_vec, _ = PsiMN.psi_m_n(self.lower.event_2d_cells, 1, self.m * self.n)
        self.upper.op_vec, _ = PsiMN.psi_m_n(self.upper.event_2d_cells, 1, self.m * self.n)

        self.op_name = "psi_" + str(self.m) + str(self.n)
        self.upper.op_name = "upper_psi_1" + str(self.n * self.m)
        self.lower.op_name = "lower_psi_1" + str(self.n * self.m)

    def calc_write(self, calc_correlation=True, bin_width=0.1):
        super().calc_write(calc_correlation, bin_width)
        self.lower.calc_write(calc_correlation, bin_width)
        self.upper.calc_write(calc_correlation, bin_width)
        np.savetxt(os.path.join(os.path.join(self.sim_path, "OP"), "lower_" + str(self.spheres_ind) + ".txt"),
                   self.lower.event_2d_cells.all_centers)
        np.savetxt(os.path.join(os.path.join(self.sim_path, "OP"), "upper_" + str(self.spheres_ind) + ".txt"),
                   self.upper.event_2d_cells.all_centers)