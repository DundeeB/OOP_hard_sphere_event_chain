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

    def correlation(self, bin_width=0.1):
        sp = self.event_2d_cells.all_spheres
        if self.op_vec is None: self.calc_order_parameter()

        # pairs = [(i, j) for j in range(self.N) for i in range(j)]
        # random.shuffle(pairs)
        # pairs = pairs[:int(n_pairs)]
        # pairs_dr = [Metric.cyclic_dist(self.event_2d_cells.boundaries, sp[j], sp[k]) for (j, k) in pairs]
        # pairs_dr = [np.linalg.norm(np.array(sp[j].center) - sp[k].center) for (j, k) in pairs]
        # phiphi_vec = [self.op_vec[j] * np.conj(self.op_vec[k]) for (j, k) in pairs]

        phiphi_vec = (np.conj(np.transpose(np.matrix(self.op_vec))) *
                      np.matrix(self.op_vec)).reshape((len(self.op_vec) ** 2,))
        x = np.array([r[0] for r in self.event_2d_cells.all_centers])
        y = np.array([r[1] for r in self.event_2d_cells.all_centers])
        dx = (x.reshape((len(x), 1)) - x.reshape((1, len(x)))).reshape(len(x) ** 2, )
        dy = (y.reshape((len(y), 1)) - y.reshape((1, len(y)))).reshape(len(y) ** 2, )
        # lx, ly = self.event_2d_cells.boundaries.edges[:2]
        # for i in range(len(dx)):
        #     v = [dx[i], dx[i]+lx, dx[i]-lx]
        #     dx[i] = v[np.argmin(np.abs(v))]
        #     v = [dy[i], dy[i] + ly, dy[i] - ly]
        #     dy[i] = v[np.argmin(np.abs(v))]
        pairs_dr = np.sqrt(dx ** 2 + dy ** 2)
        # I = np.argsort(pairs_dr)
        # pairs_dr = pairs_dr[I]
        # phiphi_vec = phiphi_vec[I]

        centers = np.linspace(0, np.max(pairs_dr), int(np.max(pairs_dr) / bin_width) + 1) + bin_width / 2
        counts = np.zeros(len(centers))
        phiphi_hist = np.zeros(len(centers), dtype=np.complex)
        for i, c in enumerate(centers):
            I = np.where(np.logical_and(pairs_dr > c - bin_width / 2, pairs_dr < c + bin_width / 2))[0]
            phiphi_hist[i] = np.sum(phiphi_vec[0, I])
            counts[i] = len(I)
        I = np.where(np.logical_or(counts != 0, phiphi_hist != np.nan))
        counts = counts[I]
        phiphi_hist = np.real(phiphi_hist[I]) / counts[I] + 1j * np.imag(phiphi_hist[I]) / counts[I]
        centers = centers[I]
        return phiphi_hist, centers, counts, phiphi_vec, pairs_dr

    def calc_write(self, calc_correlation=True, bin_width=0.1):
        if self.op_vec is None: self.calc_order_parameter()
        f = lambda a, b: os.path.join(a, b)
        if not os.path.exists(f(self.sim_path, "OP")): os.mkdir(f(self.sim_path, "OP"))
        g = lambda name, vec: np.savetxt(
            f(f(self.sim_path, "OP"), self.op_name + "_" + name + "_" + str(self.spheres_ind)) + ".txt", vec)
        g("vec", self.op_vec)
        if not calc_correlation: return

        phi_hist, centers, counts, phiphi, pairs_dr = self.correlation(bin_width=bin_width)
        g("corrhist", phi_hist)
        g("centershist", centers)
        g("countshist", counts)
        # g("drpairs", pairs_dr)
        # g("corrpairs", phiphi)


class PsiMN(OrderParameter):

    def __init__(self, sim_path, m, n):
        super().__init__(sim_path)
        self.m, self.n = m, n

    def calc_order_parameter(self):
        centers = self.event_2d_cells.all_centers
        sp = self.event_2d_cells.all_spheres
        cyc_bound = CubeBoundaries(self.event_2d_cells.boundaries.edges[:2], 2 * [BoundaryType.CYCLIC])
        cyc = lambda p1, p2: Metric.cyclic_dist(cyc_bound, Sphere(p1, 1), Sphere(p2, 1))
        self.graph = kneighbors_graph([p[:2] for p in centers], n_neighbors=self.n, metric=cyc)
        psimn_vec = np.zeros(self.N, dtype=np.complex)
        for i in range(self.N):
            sp[i].nearest_neighbors = [sp[j] for j in self.graph.getrow(i).indices]
            dr = [np.array(centers[i]) - s.center for s in sp[i].nearest_neighbors]
            t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
            psi_n = np.mean(np.exp(1j * self.n * t))
            psimn_vec[i] = np.abs(psi_n) * np.exp(1j * self.m * np.angle(psi_n))
        self.op_vec = psimn_vec
        self.op_name = "psi_" + str(self.m) + str(self.n)
        return psimn_vec
