#!/Local/ph_daniel/anaconda3/bin/python -u
import numpy as np
from SnapShot import *
from Structure import *
from EventChainActions import *
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import single_source_shortest_path_length
import os
import sys
import random
from datetime import date
from scipy.spatial import Delaunay
from scipy.optimize import fmin
import scipy.sparse
import time

epsilon = 1e-8
day = 86400  # sec


# TODO: local rho

class OrderParameter:

    def __init__(self, sim_path, centers=None, spheres_ind=None, calc_upper_lower=False, vec_name="vec",
                 correlation_name="correlation", **kwargs):
        self.sim_path = sim_path
        self.write_or_load = WriteOrLoad(sim_path)
        self.update_centers(centers, spheres_ind)
        self.op_vec = None
        self.op_corr = None
        self.corr_centers = None
        self.counts = None
        self.vec_name = vec_name
        self.correlation_name = correlation_name
        self.op_father_dir = os.path.join(self.sim_path, "OP")
        if calc_upper_lower:
            upper_centers = [c for c in self.spheres if c[2] >= self.event_2d_cells.boundaries[2] / 2]
            lower_centers = [c for c in self.spheres if c[2] < self.event_2d_cells.boundaries[2] / 2]
            self.upper = type(self)(sim_path, centers=upper_centers, spheres_ind=self.spheres_ind,
                                    calc_upper_lower=False, **kwargs)
            self.lower = type(self)(sim_path, centers=lower_centers, spheres_ind=self.spheres_ind,
                                    calc_upper_lower=False, **kwargs)
            self.upper.op_name = "upper_" + self.op_name
            self.lower.op_name = "lower_" + self.op_name

    @property
    def op_dir_path(self):
        return os.path.join(self.op_father_dir, self.op_name)

    @property
    def mean_vs_real_path(self):
        return os.path.join(self.op_dir_path, 'mean_vs_real.txt')

    @property
    def vec_path(self):
        return os.path.join(self.op_dir_path, self.vec_name + "_" + str(self.spheres_ind) + '.txt')

    @property
    def corr_path(self):
        return os.path.join(self.op_dir_path, self.correlation_name + "_" + str(self.spheres_ind) + '.txt')

    def update_centers(self, centers, spheres_ind):
        if centers is None or spheres_ind is None:
            centers, spheres_ind = self.write_or_load.last_spheres()
        self.spheres_ind = spheres_ind
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = self.write_or_load.load_Input()
        self.event_2d_cells = Event2DCells(edge, n_row, n_col, l_z)
        self.event_2d_cells.append_sphere([Sphere(c, rad) for c in centers])
        self.event_2d_cells.update_all_spheres()
        self.spheres = self.event_2d_cells.all_centers
        self.N = len(centers)

    def calc_order_parameter(self, calc_upper_lower=False):
        """to be override by child class"""
        pass

    def correlation(self, bin_width=0.1, calc_upper_lower=False, low_memory=True, randomize=False,
                    realizations=int(1e7), time_limit=2 * day):
        if self.op_vec is None: self.calc_order_parameter()
        lx, ly = self.event_2d_cells.boundaries[:2]
        l = np.sqrt(lx ** 2 + ly ** 2) / 2
        centers = np.linspace(0, np.ceil(l / bin_width) * bin_width, int(np.ceil(l / bin_width)) + 1) + bin_width / 2
        kmax = len(centers) - 1
        counts = np.zeros(len(centers))
        phiphi_hist = np.zeros(len(centers))
        init_time = time.time()
        N = len(self.spheres)
        if low_memory:
            if randomize:
                for realization in range(realizations):
                    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
                    phi_phi, k = self.__pair_corr__(i, j, bin_width)
                    k = int(min(k, kmax))
                    counts[k] += 2
                    phiphi_hist[k] += 2 * np.real(phi_phi)
                    if realization % 1000 == 0 and time.time() - init_time > time_limit:
                        break
            else:
                for i in range(N):
                    for j in range(i):  # j<i, j=i not interesting and j>i double counting accounted for in counts
                        phi_phi, k = self.__pair_corr__(i, j, bin_width)
                        k = int(min(k, kmax))
                        counts[k] += 2  # r-r' and r'-r
                        phiphi_hist[k] += 2 * np.real(phi_phi)  # a+a'=2Re(a)
                realization = N * (N - 1) / 2
        else:
            N = len(self.op_vec)
            v = np.array(self.op_vec).reshape(1, N)
            phiphi_vec = (np.conj(v) * v.T).reshape((N ** 2,))
            x = np.array([r[0] for r in self.spheres])
            y = np.array([r[1] for r in self.spheres])
            dx = (x.reshape((len(x), 1)) - x.reshape((1, len(x)))).reshape(len(x) ** 2, )
            dy = (y.reshape((len(y), 1)) - y.reshape((1, len(y)))).reshape(len(y) ** 2, )
            dx = np.minimum(np.abs(dx), np.minimum(np.abs(dx + lx), np.abs(dx - lx)))
            dy = np.minimum(np.abs(dy), np.minimum(np.abs(dy + ly), np.abs(dy - ly)))
            pairs_dr = np.sqrt(dx ** 2 + dy ** 2)

            I = np.argsort(pairs_dr)
            pairs_dr = pairs_dr[I]
            phiphi_vec = phiphi_vec[0, I]
            i = 0
            for j in range(len(pairs_dr)):
                if pairs_dr[j] > centers[i] + bin_width / 2:
                    i += 1
                phiphi_hist[i] += np.real(phiphi_vec[0, j])
                counts[i] += 1
            realization = N ** 2
        print("\nTime Passed: " + str((time.time() - init_time) / day) + " days.\nSummed " + str(
            realization) + " pairs")
        self.counts = counts
        self.op_corr = phiphi_hist / counts
        self.corr_centers = centers

        if calc_upper_lower:
            self.lower.correlation(bin_width, low_memory=low_memory, randomize=randomize, realizations=realizations)
            self.upper.correlation(bin_width, low_memory=low_memory, randomize=randomize, realizations=realizations)

    def __pair_corr__(self, i, j, bin_width):
        lx, ly = self.event_2d_cells.boundaries[:2]
        r, r_ = self.spheres[i], self.spheres[j]
        dr_vec = np.array(r) - r_
        dx = np.min(np.abs([dr_vec[0], dr_vec[0] + lx, dr_vec[0] - lx]))
        dy = np.min(np.abs([dr_vec[1], dr_vec[1] + ly, dr_vec[1] - ly]))
        dr = np.sqrt(dx ** 2 + dy ** 2)
        k = int(np.floor(dr / bin_width))
        return self.op_vec[i] * np.conjugate(self.op_vec[j]), k

    def write(self, write_correlations=True, write_vec=False, write_upper_lower=False):
        if not os.path.exists(self.op_father_dir): os.mkdir(self.op_father_dir)
        if not os.path.exists(self.op_dir_path): os.mkdir(self.op_dir_path)
        if write_vec:
            if self.op_vec is None: raise (Exception("Should calculate vec before writing"))
            np.savetxt(self.vec_path, self.op_vec)
        if write_correlations:
            if self.op_corr is None: raise (Exception("Should calculate correlation before writing"))
            np.savetxt(self.corr_path, np.transpose([self.corr_centers, self.op_corr, self.counts]))
        if write_upper_lower:
            self.lower.write(write_correlations, write_vec, write_upper_lower=False)
            self.upper.write(write_correlations, write_vec, write_upper_lower=False)

    @staticmethod
    def exists(file_path):
        if not os.path.exists(file_path): return False
        A = np.loadtxt(file_path, dtype=complex)  # complex most general so this part would not raise error
        if A is None: return False
        if len(A) == 0: return False
        return True

    def read_vec(self):
        self.op_vec = np.loadtxt(self.vec_path, dtype=complex)

    def read_or_calc_write(self, **calc_order_parameter_args):
        if OrderParameter.exists(self.vec_path):
            self.read_vec()
        else:
            self.calc_order_parameter(**calc_order_parameter_args)
            self.write(write_correlations=False, write_vec=True)

    def calc_for_all_realizations(self, calc_mean=True, calc_correlations=True, **correlation_kwargs):
        init_time = time.time()
        op_father_dir = os.path.join(self.sim_path, "OP")
        if not os.path.exists(op_father_dir): os.mkdir(op_father_dir)
        op_dir = os.path.join(op_father_dir, self.op_name)
        if not os.path.exists(op_dir): os.mkdir(op_dir)
        mean_vs_real_reals, mean_vs_real_mean = [], []
        if os.path.exists(self.mean_vs_real_path):
            mat = np.loadtxt(self.mean_vs_real_path, dtype=complex)
            if not mat.shape == (2,):
                mean_vs_real_reals = [int(np.real(r)) for r in mat[:, 0]]
                mean_vs_real_mean = [p for p in mat[:, 1]]
        i = 0
        realizations = self.write_or_load.realizations()
        realizations.append(0)

        while time.time() - init_time < 2 * day and i < len(realizations):
            sp_ind = realizations[i]
            if sp_ind != 0:
                centers = np.loadtxt(os.path.join(self.sim_path, str(sp_ind)))
            else:
                centers = np.loadtxt(os.path.join(self.sim_path, 'Initial Conditions'))
            self.update_centers(centers, sp_ind)
            if (type(self) is not PositionalCorrelationFunction):
                self.read_or_calc_write()
            if calc_mean and (sp_ind not in mean_vs_real_reals):
                mean_vs_real_reals.append(sp_ind)
                mean_vs_real_mean.append(np.mean(self.op_vec))
                I = np.argsort(mean_vs_real_reals)
                sorted_reals = np.array(mean_vs_real_reals)[I]
                sorted_mean = np.array(mean_vs_real_mean)[I]
                np.savetxt(self.mean_vs_real_path, np.array([sorted_reals, sorted_mean]).T)
            if calc_correlations and (not OrderParameter.exists(self.corr_path)):
                self.correlation(**correlation_kwargs)
                self.write(write_correlations=True, write_vec=False)
            i += 1


class Graph(OrderParameter):
    def __init__(self, sim_path, k_nearest_neighbors, directed=False, centers=None, spheres_ind=None,
                 calc_upper_lower=False, **kwargs):
        single_layer_k = 4 if k_nearest_neighbors == 4 else 6
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower, k_nearest_neighbors=single_layer_k, **kwargs)
        # extra argument k_nearest_neighbors goes to upper and lower layers
        self.k = k_nearest_neighbors
        self.directed = directed
        self.graph_father_path = os.path.join(self.op_father_dir, "Graph")

    @property
    def direc_str(self):
        return "k=" + str(self.k) + ("_directed" if self.directed else "_undirected")

    @property
    def graph_file_path(self):
        return os.path.join(self.graph_father_path, self.direc_str + "_" + str(self.spheres_ind) + ".npz")

    def calc_graph(self):
        if not os.path.exists(self.graph_father_path): os.mkdir(self.graph_father_path)
        recalc_graph = True
        if os.path.exists(self.graph_file_path):
            recalc_graph = False
            self.graph = scipy.sparse.load_npz(self.graph_file_path)
            if self.graph.shape != (self.N, self.N):
                recalc_graph = True
        if recalc_graph:
            cast_sphere = lambda c, r=1, z=0: Sphere([x for x in c] + [z], r)
            cyc = lambda p1, p2: Metric.cyclic_dist(self.event_2d_cells.boundaries, cast_sphere(p1), cast_sphere(p2))
            self.graph = kneighbors_graph([p[:2] for p in self.spheres], n_neighbors=self.k, metric=cyc)
            if not self.directed:
                I, J, _ = scipy.sparse.find(self.graph)[:]
                Ed = [(i, j) for (i, j) in zip(I, J)]
                Eud = []
                udgraph = scipy.sparse.csr_matrix((self.N, self.N))
                for i, j in Ed:
                    if ((j, i) in Ed) and ((i, j) not in Eud) and ((j, i) not in Eud):
                        Eud.append((i, j))
                        udgraph[i, j] = 1
                        udgraph[j, i] = 1
                self.graph = udgraph
            scipy.sparse.save_npz(self.graph_file_path, self.graph)
        self.nearest_neighbors = [[j for j in self.graph.getrow(i).indices] for i in range(self.N)]
        self.bonds_num = 0
        for i in range(self.N):
            self.bonds_num += len(self.nearest_neighbors[i])
        self.bonds_num /= 2


class PsiMN(Graph):

    def __init__(self, sim_path, m, n, centers=None, spheres_ind=None, calc_upper_lower=False):
        super().__init__(sim_path, k_nearest_neighbors=n, directed=True, centers=centers, spheres_ind=spheres_ind,
                         calc_upper_lower=calc_upper_lower, m=1, n=m * n)
        # extra args m,n goes to upper and lower layers
        self.m, self.n = m, n
        self.op_name = "psi_" + str(m) + str(n)

    def calc_order_parameter(self, calc_upper_lower=False):
        self.calc_graph()
        event_2d_cells, n, centers, graph = self.event_2d_cells, self.n, self.event_2d_cells.all_centers, self.graph
        psimn_vec = np.zeros(len(centers), dtype=np.complex)
        for i in range(len(centers)):
            dr = [Metric.cyclic_vec(event_2d_cells.boundaries, PsiMN.cast_sphere(centers[i]),
                                    PsiMN.cast_sphere(centers[j])) for j in self.nearest_neighbors[i]]
            t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
            psi_n = np.mean(np.exp(1j * n * t))
            psimn_vec[i] = np.abs(psi_n) * np.exp(1j * self.m * np.angle(psi_n))
        self.op_vec = psimn_vec
        if calc_upper_lower:
            self.lower.calc_order_parameter()
            self.upper.calc_order_parameter()

    def rotate_spheres(self, calc_spheres=True):
        n_orientation = self.m * self.n
        psi_avg = np.mean(self.op_vec)
        orientation = np.imag(np.log(psi_avg)) / n_orientation
        if not calc_spheres:
            return orientation
        else:
            R = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0],
                          [0.0, 0.0, 1.0]])  # rotate back from orientation-->0
            return orientation, [np.matmul(R, r) for r in self.spheres]


class PositionalCorrelationFunction(OrderParameter):

    def __init__(self, sim_path, m, n, rect_width=0.1, centers=None, spheres_ind=None, calc_upper_lower=False):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower, rect_width=rect_width)
        self.rect_width = rect_width
        self.m, self.n = m, n
        self.op_name = "pos"

    def correlation(self, bin_width=0.1, calc_upper_lower=False, low_memory=True, randomize=False,
                    realizations=int(1e7), time_limit=2 * day):
        psi = PsiMN(self.sim_path, self.m, self.n, centers=self.spheres, spheres_ind=self.spheres_ind)
        psi.calc_order_parameter()
        self.theta = psi.rotate_spheres(calc_spheres=False)
        self.correlation_name = "correlation_theta=" + str(self.theta)
        theta, rect_width = self.theta, self.rect_width
        v_hat = np.array([np.cos(theta), np.sin(theta)])
        lx, ly = self.event_2d_cells.boundaries[:2]
        l = np.sqrt(lx ** 2 + ly ** 2) / 2
        bins_edges = np.linspace(0, np.ceil(l / bin_width) * bin_width, int(np.ceil(l / bin_width)) + 1)
        kmax = len(bins_edges) - 1
        self.corr_centers = bins_edges[:-1] + bin_width / 2
        self.counts = np.zeros(len(self.corr_centers))
        init_time = time.time()
        N = len(self.spheres)
        if low_memory:
            if randomize:
                for realization in range(realizations):
                    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
                    if i == j: continue
                    k = self.__pair_dist__(self.spheres[i], self.spheres[j], v_hat, rect_width, bin_width)
                    if k is not None:
                        k = int(min(k, kmax))
                        self.counts[k] += 1
                    if realization % 1000 == 0 and time.time() - init_time > time_limit:
                        break
            else:
                for i in range(N):
                    for j in range(N):
                        if j == i: continue
                        k = self.__pair_dist__(self.spheres[i], self.spheres[j], v_hat, rect_width, bin_width)
                        if k is not None:
                            k = int(min(k, kmax))
                            self.counts[k] += 1
                realization = N * (N - 1)
        else:
            x = np.array([r[0] for r in self.spheres])
            y = np.array([r[1] for r in self.spheres])
            N = len(x)
            dx = (x.reshape((N, 1)) - x.reshape((1, N))).reshape(N ** 2, )
            dy = (y.reshape((N, 1)) - y.reshape((1, N))).reshape(N ** 2, )
            A = np.transpose([dx, dx + lx, dx - lx])
            I = np.argmin(np.abs(A), axis=1)
            J = [i for i in range(len(I))]
            dx = A[J, I]
            A = np.transpose([dy, dy + ly, dy - ly])
            I = np.argmin(np.abs(A), axis=1)
            dy = A[J, I]

            pairs_dr = np.transpose([dx, dy])

            m = lambda A, B: np.matmul(A, B)
            dist_vec = m(v_hat.reshape(2, 1), m(pairs_dr, v_hat).reshape(1, N)).T - pairs_dr
            dist_to_line = np.linalg.norm(dist_vec, axis=1)
            I = np.where(dist_to_line <= rect_width / 2)[0]
            pairs_dr = pairs_dr[I]
            J = np.where(m(pairs_dr, v_hat) > 0)[0]
            pairs_dr = pairs_dr[J]
            rs = m(pairs_dr, v_hat)
            self.counts, _ = np.histogram(rs, bins_edges)
            realization = N ** 2
        print("\nTime Passed: " + str((time.time() - init_time) / day) + " days.\nSummed " + str(
            realization) + " pairs")

        # normalize counts
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = self.write_or_load.load_Input()
        rho2D = self.N / (l_x * l_y)
        # counts --> counts*N*(N-1)/realization-->counts*(N*(N-1)/realization)*(1/(rho*a*N))
        self.op_corr = self.counts / (rho2D * bin_width * rect_width * realization / (self.N - 1))

        if calc_upper_lower:
            assert self.upper is not None, \
                "Failed calculating upper positional correlation because it was not initialized"
            self.upper.correlation(bin_width=bin_width, low_memory=low_memory, randomize=randomize,
                                   realizations=realizations)
            assert self.upper is not None, \
                "Failed calculating lower positional correlation because it was not initialized"
            self.lower.correlation(bin_width=bin_width, low_memory=low_memory, randomize=randomize,
                                   realizations=realizations)

    def __pair_dist__(self, r, r_, v_hat, rect_width, bin_width):
        lx, ly = self.event_2d_cells.boundaries[:2]
        dr = np.array(r) - r_
        dxs = [dr[0], dr[0] + lx, dr[0] - lx]
        dx = dxs[np.argmin(np.abs(dxs))]
        dys = [dr[1], dr[1] + ly, dr[1] - ly]
        dy = dys[np.argmin(np.abs(dys))]
        dr = np.array([dx, dy])
        dist_on_line = float(np.dot(dr, v_hat))
        dist_vec = v_hat * dist_on_line - dr
        dist_to_line = np.linalg.norm(dist_vec)
        if dist_to_line <= rect_width / 2 and dist_on_line > 0:
            k = int(np.floor(dist_on_line / bin_width))
            return k
        else:
            return None


class BurgerField(OrderParameter):

    def __init__(self, sim_path, centers=None, spheres_ind=None, calc_upper_lower=False):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower=False)
        self.op_name = "burger_vectors"
        if calc_upper_lower:
            upper_centers = [c for c in self.spheres if c[2] >= self.event_2d_cells.boundaries[2] / 2]
            lower_centers = [c for c in self.spheres if c[2] < self.event_2d_cells.boundaries[2] / 2]
            self.upper = BurgerField(sim_path, centers=upper_centers, spheres_ind=self.spheres_ind)
            self.lower = BurgerField(sim_path, centers=lower_centers, spheres_ind=self.spheres_ind)
            self.upper.op_name = "upper_" + self.op_name
            self.lower.op_name = "lower_" + self.op_name

    def calc_order_parameter(self, calc_upper_lower=False):
        psi = PsiMN(self.sim_path, 1, 4, centers=self.spheres, spheres_ind=self.spheres_ind)
        psi.read_or_calc_write()
        orientation = psi.rotate_spheres(calc_spheres=False)
        R = np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])

        bragg = BraggStructure(self.sim_path, 1, 4, self.spheres, self.spheres_ind)
        bragg.read_or_calc_write(psi=psi)
        a = 2 * np.pi / np.linalg.norm(bragg.k_peak)
        a1, a2 = np.array([a, 0]), np.array([0, a])

        perfect_lattice_vectors = np.array([n * a1 + m * a2 for n in range(-3, 3) for m in range(-3, 3)])
        perfect_lattice_vectors = np.array([np.matmul(R, p.T) for p in perfect_lattice_vectors])
        # rotate = lambda ps: np.matmul(R(-orientation), np.array(ps).T).T
        disloc_burger, disloc_location = BurgerField.calc_burger_vector(self.event_2d_cells, perfect_lattice_vectors)
        self.op_vec = np.concatenate((np.array(disloc_location).T, np.array(disloc_burger).T)).T  # x, y, bx, by field
        if calc_upper_lower:
            self.lower.calc_order_parameter()
            self.upper.calc_order_parameter()

    @staticmethod
    def calc_burger_vector(event_2d_cells, perfect_lattice_vectors):
        """
        Calculate the burger vector on each plaquette of the Delaunay triangulation using methods in:
            [1]	https://link.springer.com/content/pdf/10.1007%2F978-3-319-42913-7_20-1.pdf
            [2]	https://www.sciencedirect.com/science/article/pii/S0022509614001331?via%3Dihub
        :param event_2d_cells: Structure containing spheres centers, boundaries ext.
        :param wraped_psi: orientational order parameter for local orientation. psi(i) correspond to the i'th sphere
        :param perfect_lattice_vectors: list of vectors of the perfect lattice. Their magnitude is no important.
        :return: The positions (r) and burger vector at each position b. The position of a dislocation is take as the
                center of the plaquette.
        """
        wraped_centers = BurgerField.wrap_with_boundaries(event_2d_cells, w=5)
        # all spheres within w distance from cyclic boundary will be mirrored
        tri = Delaunay(wraped_centers)
        dislocation_burger = []
        dislocation_location = []
        for i, simplex in enumerate(tri.simplices):
            rc = np.mean(tri.points[simplex], 0)
            if rc[0] < 0 or rc[0] > event_2d_cells.boundaries[0] or rc[1] < 0 or rc[1] > \
                    event_2d_cells.boundaries[1]:
                continue
            b_i = BurgerField.burger_calculation(tri.points[simplex], perfect_lattice_vectors)
            if np.linalg.norm(b_i) > epsilon:
                dislocation_location.append(rc)
                dislocation_burger.append(b_i)
        return dislocation_burger, dislocation_location

    @staticmethod
    def burger_calculation(simplex_points, perfect_lattice_vectors):
        simplex_points = np.array(simplex_points)

        ts = []
        rc = np.mean(simplex_points, 0)
        for p in simplex_points:
            ts.append(np.arctan2(p[0] - rc[0], p[1] - rc[1]))
        I = np.argsort(ts)
        simplex_points = simplex_points[I]  # calculate burger circuit always anti-clockwise
        which_L = lambda x_ab: np.argmin([np.linalg.norm(x_ab - L_) for L_ in perfect_lattice_vectors])
        L_ab = lambda x_ab: perfect_lattice_vectors[which_L(x_ab)]
        Ls = [L_ab(simplex_points[b] - simplex_points[a]) for (a, b) in [(0, 1), (1, 2), (2, 0)]]
        return np.sum(Ls, 0)

    @staticmethod
    def wrap_with_boundaries(event_2d_cells, w):
        centers = np.array(event_2d_cells.all_centers)[:, :2]
        Lx, Ly = event_2d_cells.boundaries[:2]
        x = centers[:, 0]
        y = centers[:, 1]

        sp1 = centers[np.logical_and(x - Lx > -w, y < w), :] + [-Lx, Ly]
        sp2 = centers[y < w, :] + [0, Ly]
        sp3 = centers[np.logical_and(x < w, y < w), :] + [Lx, Ly]
        sp4 = centers[x - Lx > -w, :] + [-Lx, 0]
        sp5 = centers[:, :] + [0, 0]
        sp6 = centers[x < w, :] + [Lx, 0]
        sp7 = centers[np.logical_and(x - Lx > -w, y - Ly > -w), :] + [-Lx, -Ly]
        sp8 = centers[y - Ly > -w, :] + [0, -Ly]
        sp9 = centers[np.logical_and(x < w, y - Ly > -w), :] + [Lx, -Ly]

        wraped_centers = np.concatenate((sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9))
        return wraped_centers


class BraggStructure(OrderParameter):
    def __init__(self, sim_path, m, n, centers=None, spheres_ind=None):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower=False)
        self.op_name = "Bragg_S"
        self.k_peak = None
        self.data = []
        self.m, self.n = m, n

    def calc_eikr(self, k):
        return np.exp([1j * (k[0] * r[0] + k[1] * r[1]) for r in self.spheres])

    def S(self, k):
        sum_r = np.sum(self.calc_eikr(k))
        N = len(self.spheres)
        S_ = np.real(1 / N * sum_r * np.conjugate(sum_r))
        self.data.append([k[0], k[1], S_])
        return S_

    def tour_on_circle(self, k_radii, theta=None):
        if theta is None:
            theta_peak = np.arctan2(self.k_peak[1], self.k_peak[0])
            theta = np.mod(theta_peak + np.linspace(0, 1, 101) * 2 * np.pi, 2 * np.pi)
            theta = np.sort(np.concatenate([theta, [np.pi / 4 * x for x in range(8)]]))
        for t in theta:
            self.S(k_radii * np.array([np.cos(t), np.sin(t)]))

    def k_perf(self):
        # rotate self.spheres by orientation before so peak is at for square [1, 0]
        l = np.sqrt(self.event_2d_cells.l_x * self.event_2d_cells.l_y / len(self.spheres))
        if self.m == 1 and self.n == 4:
            return 2 * np.pi / l * np.array([1, 0])
        if self.m == 1 and self.n == 6:
            a = np.sqrt(2.0 / np.sqrt(3)) * l
            return 2 * np.pi / a * np.array([1.0, -1.0 / np.sqrt(3)])
        if self.m == 2 and self.n == 3:
            a = np.sqrt(4.0 / np.sqrt(3)) * l
            return 2 * np.pi / a * np.array([0, 2])
        raise NotImplementedError

    def calc_peak(self):
        S = lambda k: -self.S(k)
        self.k_peak, S_peak_m, _, _, _ = fmin(S, self.k_perf(), xtol=0.01 / len(self.spheres), ftol=1.0,
                                              full_output=True)
        self.S_peak = -S_peak_m

    def peaks_other_angles(self):
        mn = self.m * self.n
        theta0 = np.arctan2(self.k_perf()[1], self.k_perf()[0])  # k_perf is overwritten in magnetic bragg
        return [theta0 + n * 2 * np.pi / mn for n in range(1, mn)]

    def calc_other_peaks(self):
        S = lambda k: -self.S(k)
        self.other_peaks_k = [self.k_peak]
        self.other_peaks_S = [self.S_peak]
        k_radii = np.linalg.norm(self.k_perf())  # k_perf is overwritten in magnetic bragg
        for theta in self.peaks_other_angles():
            k_perf = k_radii * np.array([np.cos(theta), np.sin(theta)])
            k_peak, S_peak_m, _, _, _ = fmin(S, k_perf, xtol=0.01 / len(self.spheres), ftol=1.0, full_output=True)
            self.other_peaks_k.append(k_peak)
            self.other_peaks_S.append(S_peak_m)

    def write(self, write_vec=True, write_correlations=True):
        op_vec = self.op_vec
        self.op_vec = np.array(self.data)
        # overrides the usless e^(ikr) vector for writing the important data in self.data. self.op_corr has already been
        # calculated.
        super().write(write_correlations=write_correlations, write_vec=write_vec, write_upper_lower=False)
        self.op_vec = op_vec

    def calc_order_parameter(self, psi=None):
        if psi is None:
            psi = PsiMN(self.sim_path, self.m, self.n, centers=self.spheres, spheres_ind=self.spheres_ind)
            psi.read_or_calc_write()
        _, self.spheres = psi.rotate_spheres()
        self.calc_peak()
        self.op_vec = self.calc_eikr(self.k_peak)
        k1, k2 = np.linalg.norm(self.k_perf()), np.linalg.norm(self.k_peak)
        m1, m2 = min([k1, k2]), max([k1, k2])
        dm = m2 - m1
        for k_radii in np.linspace(m1 - 10 * dm, m2 + 10 * dm, 12):
            self.tour_on_circle(k_radii)
        self.calc_other_peaks()

    def correlation(self, bin_width=0.1, low_memory=True, randomize=False, realizations=int(1e7), time_limit=2 * day):
        super().correlation(bin_width, False, low_memory, randomize, realizations, time_limit)

    def read_vec(self):
        self.data = np.loadtxt(self.vec_path, dtype=complex)
        S = [d[2] for d in self.data]
        i = np.argmax(S)
        self.S_peak = S[i]
        self.k_peak = [self.data[i][0], self.data[i][1]]
        self.op_vec = self.calc_eikr(self.k_peak)


class MagneticBraggStructure(BraggStructure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_name = "Bragg_Sm"

    def calc_eikr(self, k):
        # sum_n(z_n*e^(ikr_n))
        # z in [rad,lz-rad]-->z in [-1,1]: (z-lz/2)/(lz/2-rad)
        # For z=rad we have (rad-lz/2)/(lz/2-rad)=-1
        # For z=lz-rad we have (lz-rad-lz/2)/(lz/2-rad)=1.
        rad, lz = 1.0, self.event_2d_cells.l_z
        return np.array(
            [(r[2] - lz / 2) / (lz / 2 - rad) * np.exp(1j * (k[0] * r[0] + k[1] * r[1])) for r in self.spheres])

    def k_perf(self):
        l = np.sqrt(self.event_2d_cells.l_x * self.event_2d_cells.l_y / len(self.spheres))
        if self.m == 1 and self.n == 4:
            return np.pi / l * np.array([1, 1])
        # TODO: update by k*a_AB
        if self.m == 1 and self.n == 6:
            a = np.sqrt(2.0 / np.sqrt(3)) * l
            return np.pi / a * np.array([1.0, 1.0 / np.sqrt(3)])
        if self.m == 2 and self.n == 3:
            a = np.sqrt(4.0 / np.sqrt(3)) * l
            return np.pi / a * np.array([2.0, 2.0])
        raise NotImplementedError


class MagneticTopologicalCorr(Graph):
    def __init__(self, sim_path, k_nearest_neighbors, directed=False, centers=None, spheres_ind=None,
                 calc_upper_lower=False):
        super().__init__(sim_path, k_nearest_neighbors, directed, centers, spheres_ind, calc_upper_lower)

    @property
    def op_name(self):
        return "gM_" + self.direc_str

    def calc_order_parameter(self):
        self.calc_graph()
        rad, lz = 1.0, self.event_2d_cells.l_z
        self.op_vec = [(r[2] - lz / 2) / (lz / 2 - rad) for r in self.spheres]

    def correlation(self, calc_upper_lower=False):
        N = len(self.spheres)
        kbound = N  # eassier than finding graph's diameter
        kmax = 0
        counts = np.zeros(kbound + 1)
        phiphi_hist = np.zeros(kbound + 1)
        init_time = time.time()
        for i in range(N):
            shortest_paths_from_i = single_source_shortest_path_length(self.graph, i)
            # Complicated implementation because the simple one returns NxN matrix, and another simple option of node to
            # node shortest path required additional libs and sklearn was already installed for me
            for j in range(N):
                try:
                    k = int(shortest_paths_from_i[j])
                except KeyError:  # j is not in a connected component of i
                    continue
                if k > kmax: kmax = k
                phi_phi = (-1) ** k * self.op_vec[i] * self.op_vec[j]
                counts[k] += 1
                phiphi_hist[k] += phi_phi
        realization = N ** 2
        print("\nTime Passed: " + str((time.time() - init_time) / day) + " days.\nSummed " + str(
            realization) + " pairs")
        counts = counts[:kmax + 1]
        phiphi_hist = phiphi_hist[:kmax + 1]
        self.counts = counts
        self.op_corr = phiphi_hist / counts
        self.corr_centers = np.array(range(kmax + 1))

        if calc_upper_lower:
            self.lower.correlation()
            self.upper.correlation()


class Ising(Graph):
    def __init__(self, sim_path, k_nearest_neighbors, directed=False, centers=None, spheres_ind=None, J=None):
        super().__init__(sim_path, k_nearest_neighbors, directed, centers, spheres_ind, vec_name="ground_state",
                         correlation_name="E_vs_J")
        l_y = self.event_2d_cells.boundaries[2]
        self.z_spins = [(1 if p[2] > l_y / 2 else -1) for p in self.spheres]
        self.calc_graph()
        self.J = J

    @property
    def op_name(self):
        return "Ising_" + self.direc_str

    def initialize(self, random_initialization=True, J=None):
        if J is not None:
            self.J = J
        self.op_vec = [((2 * random.randint(0, 1) - 1) if random_initialization else z) for z in self.z_spins]
        self.calc_EM()

    def calc_EM(self):
        self.E = 0
        for i in range(self.N):
            for j in self.nearest_neighbors[i]:
                self.E -= self.J / 2 * self.op_vec[i] * self.op_vec[j]  # double counting bonds
        self.M = 0  # Magnetization is if op_vec, ising spins, is corr or anti corr to up down initial partition
        for s, z in zip(self.op_vec, self.z_spins):
            self.M += s * z

    def Metropolis_flip(self, i=None):
        if i is None:
            i = random.randint(0, self.N - 1)
        de = 0.0
        for j in self.nearest_neighbors[i]:
            de += 2 * self.J * self.op_vec[i] * self.op_vec[j]
        A = min(1, np.exp(-de))
        u = random.random()
        if u <= A:
            self.op_vec[i] *= -1

    def anneal(self, iterations, dTditer=0, diter_save=1):
        M, E, J = [], [], []
        T = -1 / self.J
        for i in range(iterations):
            if i % diter_save == 0 or (i == iterations - 1):
                self.calc_EM()
                M.append(self.M)
                E.append(self.E)
                J.append(self.J)
            self.Metropolis_flip()
            T += dTditer
            self.J = -1 / T

        return E, J, M

    def local_freeze(self):
        current_J = self.J
        self.J = -100
        for i in range(self.N):
            self.Metropolis_flip(i)
        self.J = current_J

    def frustrated_bonds(self, E, J):
        return 1 / 2 * (1 - np.array(E) / (self.bonds_num * np.array(J)))

    def calc_order_parameter(self, J_range=(-0.5, -2), iterations=None, realizations=10, samples=1000,
                             random_initialization=True, save_annealing=True, localy_freeze=True):
        if iterations is None:
            iterations = self.N * int(4e4)
        diter_save = int(iterations / samples)
        minE = float('inf')
        minEconfig = None
        frustration, Ms = [], []
        for i in range(realizations):
            self.initialize(random_initialization=random_initialization, J=J_range[0])
            E, J, M = self.anneal(iterations, diter_save=diter_save,
                                  dTditer=-(1 / J_range[1] - 1 / J_range[0]) / iterations)
            frustration.append(self.frustrated_bonds(E, J))
            Ms.append(np.array(M) / self.N)
            mE = min(frustration[-1])
            if mE < minE:
                minE = mE
                minEconfig = self.op_vec
        self.op_vec = minEconfig
        self.local_freeze()
        annel_path = os.path.join(self.op_dir_path, "anneal_" + str(self.spheres_ind) + '.txt')
        np.savetxt(annel_path, np.transpose([J] + frustration + Ms))

    def correlation(self, J_range=(-0.1, -1), iterations=None, realizations=1, dJ=0.05):
        if iterations is None:
            iterations = self.N * int(1e4)
        Jc = -1 / 2.269
        Jarr = np.linspace(J_range[0], J_range[1], int(np.abs(J_range[1] - J_range[0]) / dJ) + 1)
        Jarr = np.sort([J for J in Jarr] + [Jc])
        frustration = []
        for J in Jarr:
            E_reals = []
            for real in range(realizations):
                self.initialize(J=J)
                E, _, _ = self.anneal(iterations, diter_save=iterations, dTditer=0)
                E_reals.append(E[-1])
            E = np.mean(E_reals)
            frustration.append(self.frustrated_bonds(E, J))
        self.corr_centers = Jarr
        self.counts = Jarr * 0 + realizations
        self.op_corr = np.array(frustration)


def main(sim_name, calc_type):
    correlation_kwargs = {'randomize': False, 'time_limit': 2 * day}

    prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results3.0/"
    sim_path = os.path.join(prefix, sim_name)
    op_dir = os.path.join(sim_path, "OP")
    if not os.path.exists(op_dir): os.mkdir(op_dir)
    log = os.path.join(op_dir, "log")
    sys.stdout = open(log, "a")
    if calc_type.endswith("23"):
        m, n = 2, 3
    if calc_type.endswith("14"):
        m, n = 1, 4
    if calc_type.endswith("16"):
        m, n = 1, 6
    calc_correlations, calc_mean = True, True
    if calc_type.startswith("psi"):
        op = PsiMN(sim_path, m, n)
        if calc_type.startswith("psi_mean"):
            calc_correlations = False
    if calc_type.startswith("pos"):
        op = PositionalCorrelationFunction(sim_path, m, n)
        calc_mean = False
    if calc_type == "burger_square":
        op = BurgerField(sim_path)
        calc_mean, calc_correlations = False, False
    if calc_type.startswith("Bragg_S"):
        if calc_type.startswith("Bragg_Sm"):
            op = MagneticBraggStructure(sim_path, m, n)
        else:
            op = BraggStructure(sim_path, m, n)
    if calc_type.startswith("gM"):
        op = MagneticTopologicalCorr(sim_path, k_nearest_neighbors=n)
        calc_mean = False
        correlation_kwargs = {}
    if calc_type.startswith('Ising'):
        op = Ising(sim_path, k_nearest_neighbors=n)
        calc_mean = False
        correlation_kwargs = {}
    print(
        "\n\n\n-----------\nDate: " + str(date.today()) + "\nType: " + calc_type + "\nCorrelation arguments:" + str(
            correlation_kwargs) + "\nCalc correlations: " + str(calc_correlations) + "\nCalc mean: " + str(
            calc_mean),
        file=sys.stdout)
    op.calc_for_all_realizations(calc_correlations=calc_correlations, calc_mean=calc_mean, **correlation_kwargs)


if __name__ == "__main__":
    main(sim_name=sys.argv[1], calc_type=sys.argv[2])
