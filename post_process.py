#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
from SnapShot import *
from Structure import *
from EventChainActions import *
from sklearn.neighbors import *
import os
import sys
import random
import re
from datetime import date
from scipy.spatial import Delaunay


class OrderParameter:

    def __init__(self, sim_path, centers=None, spheres_ind=None, calc_upper_lower=True, **kwargs):
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
        self.op_corr = None
        self.corr_centers = None
        self.counts = None

        self.op_name = "phi"
        if calc_upper_lower:
            upper_centers = [c for c in self.event_2d_cells.all_centers if
                             c[2] >= self.event_2d_cells.boundaries.edges[2] / 2]
            lower_centers = [c for c in self.event_2d_cells.all_centers if
                             c[2] < self.event_2d_cells.boundaries.edges[2] / 2]
            self.upper = type(self)(sim_path, centers=upper_centers, spheres_ind=self.spheres_ind,
                                    calc_upper_lower=False, **kwargs)
            self.lower = type(self)(sim_path, centers=lower_centers, spheres_ind=self.spheres_ind,
                                    calc_upper_lower=False, **kwargs)

    def calc_order_parameter(self, calc_upper_lower=True):
        """to be override by child class"""
        pass

    def correlation(self, bin_width=0.2, calc_upper_lower=True, low_memory=False, randomize=False,
                    realizations=int(1e7)):
        if self.op_vec is None: self.calc_order_parameter()
        lx, ly = self.event_2d_cells.boundaries.edges[:2]
        l = np.sqrt(lx ** 2 + ly ** 2) / 2
        centers = np.linspace(0, np.ceil(l / bin_width) * bin_width, int(np.ceil(l / bin_width)) + 1) + bin_width / 2
        counts = np.zeros(len(centers))
        phiphi_hist = np.zeros(len(centers), dtype=np.complex)
        if not low_memory:
            phiphi_vec = (np.conj(np.transpose(np.matrix(self.op_vec))) *
                          np.matrix(self.op_vec)).reshape((len(self.op_vec) ** 2,))
            x = np.array([r[0] for r in self.event_2d_cells.all_centers])
            y = np.array([r[1] for r in self.event_2d_cells.all_centers])
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
                phiphi_hist[i] += phiphi_vec[0, j]
                counts[i] += 1
        else:
            if not randomize:
                for i in range(len(self.event_2d_cells.all_centers)):
                    for j in range(len(self.event_2d_cells.all_centers)):
                        phi_phi, k = self.__pair_corr__(i, j, centers, bin_width)
                        counts[k] += 1
                        phiphi_hist[k] += phi_phi
            else:
                for real in range(realizations):
                    i, j = random.randint(0, len(self.op_vec) - 1), random.randint(0, len(self.op_vec) - 1)
                    phi_phi, k = self.__pair_corr__(i, j, centers, bin_width)
                    counts[k] += 1
                    phiphi_hist[k] += phi_phi
        self.counts = counts
        self.op_corr = np.real(phiphi_hist) / counts + 1j * np.imag(phiphi_hist) / counts
        self.corr_centers = centers

        if calc_upper_lower:
            self.lower.correlation(bin_width, calc_upper_lower=False, low_memory=low_memory, randomize=randomize,
                                   realizations=realizations)
            self.upper.correlation(bin_width, calc_upper_lower=False, low_memory=low_memory, randomize=randomize,
                                   realizations=realizations)

    def __pair_corr__(self, i, j, centers, bin_width):
        lx, ly = self.event_2d_cells.boundaries.edges[:2]
        r, r_ = self.event_2d_cells.all_centers[i], self.event_2d_cells.all_centers[j]
        dr_vec = np.array(r) - r_
        dx = np.min(np.abs([dr_vec[0], dr_vec[0] + lx, dr_vec[0] - lx]))
        dy = np.min(np.abs([dr_vec[1], dr_vec[1] + ly, dr_vec[1] - ly]))
        dr = np.sqrt(dx ** 2 + dy ** 2)
        k = \
            np.where(np.logical_and(centers - bin_width / 2 <= dr, centers + bin_width / 2 > dr))[0][0]
        return self.op_vec[i] * np.conjugate(self.op_vec[j]), k

    def write(self, write_correlation=True, write_vec=False, write_upper_lower=True):
        f = lambda a, b: os.path.join(a, b)
        g = lambda name, mat: np.savetxt(
            f(f(self.sim_path, "OP"), self.op_name + "_" + name + "_" + str(self.spheres_ind)) + ".txt", mat)
        if not os.path.exists(f(self.sim_path, "OP")): os.mkdir(f(self.sim_path, "OP"))
        if write_vec:
            if self.op_vec is None: raise (Exception("Should calculate correlation before writing"))
            g("vec", self.op_vec)
        if write_correlation:
            if self.op_corr is None: raise (Exception("Should calculate correlation before writing"))
            g("correlation", np.transpose([self.corr_centers, np.abs(self.op_corr), self.counts]))
        if write_upper_lower:
            self.lower.write(write_correlation, write_vec, write_upper_lower=False)
            self.upper.write(write_correlation, write_vec, write_upper_lower=False)
            np.savetxt(os.path.join(os.path.join(self.sim_path, "OP"), "lower_" + str(self.spheres_ind) + ".txt"),
                       self.lower.event_2d_cells.all_centers)
            np.savetxt(os.path.join(os.path.join(self.sim_path, "OP"), "upper_" + str(self.spheres_ind) + ".txt"),
                       self.upper.event_2d_cells.all_centers)


class PsiMN(OrderParameter):

    def __init__(self, sim_path, m, n, centers=None, spheres_ind=None, calc_upper_lower=True):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower, m=1, n=m * n)
        # extra args m,n goes to upper and lower layers
        self.m, self.n = m, n
        self.op_name = "psi_" + str(m) + str(n)
        if calc_upper_lower:
            self.upper.op_name = "upper_psi_1" + str(n * m)
            self.lower.op_name = "lower_psi_1" + str(n * m)

    @staticmethod
    def psi_m_n(event_2d_cells, m, n):
        centers = event_2d_cells.all_centers
        cyc_bound = CubeBoundaries(event_2d_cells.boundaries.edges[:2], 2 * [BoundaryType.CYCLIC])
        cyc = lambda p1, p2: Metric.cyclic_dist(cyc_bound, Sphere(p1, 1), Sphere(p2, 1))
        graph = kneighbors_graph([p[:2] for p in centers], n_neighbors=n, metric=cyc)
        psimn_vec = np.zeros(len(centers), dtype=np.complex)
        for i in range(len(centers)):
            dr = [np.array(centers[i]) - centers[j] for j in graph.getrow(i).indices]
            t = np.arctan2([r[1] for r in dr], [r[0] for r in dr])
            psi_n = np.mean(np.exp(1j * n * t))
            psimn_vec[i] = np.abs(psi_n) * np.exp(1j * m * np.angle(psi_n))
        return psimn_vec, graph

    def calc_order_parameter(self, calc_upper_lower=True):
        self.op_vec, _ = PsiMN.psi_m_n(self.event_2d_cells, self.m, self.n)
        if calc_upper_lower:
            self.lower.calc_order_parameter(calc_upper_lower=False)
            self.upper.calc_order_parameter(calc_upper_lower=False)


class PositionalCorrelationFunction(OrderParameter):

    def __init__(self, sim_path, theta=0, rect_width=0.2, centers=None, spheres_ind=None, calc_upper_lower=True):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower, theta=theta, rect_width=rect_width)
        self.theta = theta
        self.rect_width = rect_width
        self.op_name = "positional_theta=" + str(theta)
        if calc_upper_lower:
            self.upper.op_name = "upper_" + self.op_name
            self.lower.op_name = "lower_" + self.op_name

    def correlation(self, bin_width=0.2, calc_upper_lower=True, low_memory=False, randomize=False,
                    realizations=int(1e7)):
        theta, rect_width = self.theta, self.rect_width
        v_hat = np.transpose(np.matrix([np.cos(theta), np.sin(theta)]))
        lx, ly = self.event_2d_cells.boundaries.edges[:2]
        l = np.sqrt(lx ** 2 + ly ** 2) / 2
        bins_edges = np.linspace(0, np.ceil(l / bin_width) * bin_width, int(np.ceil(l / bin_width)) + 1)
        self.corr_centers = bins_edges[:-1] + bin_width / 2
        self.counts = np.zeros(len(self.corr_centers))
        if not low_memory:
            x = np.array([r[0] for r in self.event_2d_cells.all_centers])
            y = np.array([r[1] for r in self.event_2d_cells.all_centers])
            dx = (x.reshape((len(x), 1)) - x.reshape((1, len(x)))).reshape(len(x) ** 2, )
            dy = (y.reshape((len(y), 1)) - y.reshape((1, len(y)))).reshape(len(y) ** 2, )
            A = np.transpose([dx, dx + lx, dx - lx])
            I = np.argmin(np.abs(A), axis=1)
            J = [i for i in range(len(I))]
            dx = A[J, I]
            A = np.transpose([dy, dy + ly, dy - ly])
            I = np.argmin(np.abs(A), axis=1)
            dy = A[J, I]

            pairs_dr = np.transpose([dx, dy])

            dist_vec = np.transpose(v_hat * np.transpose(pairs_dr * v_hat) - np.transpose(pairs_dr))
            dist_to_line = np.linalg.norm(dist_vec, axis=1)
            I = np.where(dist_to_line <= rect_width / 2)[0]
            pairs_dr = pairs_dr[I]
            J = np.where(pairs_dr * v_hat > 0)[0]
            pairs_dr = pairs_dr[J]
            rs = pairs_dr * v_hat
            self.counts, _ = np.histogram(rs, bins_edges)
        else:
            if not randomize:
                for r in self.event_2d_cells.all_centers:
                    for r_ in self.event_2d_cells.all_centers:
                        self.__pair_dist__(r, r_, v_hat, rect_width, bins_edges)
            else:
                for realization in range(realizations):
                    i, j = random.randint(0, len(self.event_2d_cells) - 1), random.randint(0,
                                                                                           len(self.event_2d_cells) - 1)
                    self.__pair_dist__(self.event_2d_cells[i], self.event_2d_cells[j])
        self.op_corr = self.counts / np.nanmean(self.counts[np.where(self.counts > 0)])

        if calc_upper_lower:
            assert (self.upper is not None,
                    "Failed calculating upper positional correlation because it was not initialized")
            self.upper.correlation(bin_width=bin_width, calc_upper_lower=False, low_memory=low_memory,
                                   randomize=randomize, realizations=realizations)
            assert (self.upper is not None,
                    "Failed calculating lower positional correlation because it was not initialized")
            self.lower.correlation(bin_width=bin_width, calc_upper_lower=False, low_memory=low_memory,
                                   randomize=randomize, realizations=realizations)

    def __pair_dist__(self, r, r_, v_hat, rect_width, bins_edges):
        lx, ly = self.event_2d_cells.boundaries.edges[:2]
        dr = np.array(r) - r_
        dxs = [dr[0], dr[0] + lx, dr[0] - lx]
        dx = dxs[np.argmin(np.abs(dxs))]
        dys = [dr[1], dr[1] + ly, dr[1] - ly]
        dy = dys[np.argmin(np.abs(dys))]
        dr = np.array([dx, dy])
        dist_on_line = float(np.dot(dr, v_hat))
        dist_vec = v_hat * dist_on_line - np.transpose(np.matrix(dr))
        dist_to_line = np.linalg.norm(dist_vec)
        if dist_to_line <= rect_width / 2 and dist_on_line > 0:
            k = np.where(
                np.logical_and(bins_edges[:-1] <= dist_on_line, bins_edges[1:] > dist_on_line))[0][0]
            self.counts[k] += 1


class PsiUpPsiDown(PsiMN):
    def __init__(self, sim_path, m, n, centers=None, spheres_ind=None):
        super().__init__(sim_path, m, n, centers, spheres_ind, calc_upper_lower=True)
        self.op_orig = self.op_name
        self.op_name = "psiup_psidown_" + str(m) + str(n)

    def calc_order_parameter(self):
        super().calc_order_parameter(calc_upper_lower=True)
        self.op_orig = self.op_vec


class RealizationsAveragedOP:

    def __init__(self, num_realizations, op_type, op_args):
        """

        :type sim_path: str
        :type num_realizations: int
        :param op_type: OrderParameter. Example: op_type = PsiMn
        :param op_args: Example: (sim_path,m,n,...)
        """
        self.sim_path = op_args[0]
        files = os.listdir(self.sim_path)
        numbered_files = sorted([int(f) for f in files if re.findall("^\d+$", f)])
        numbered_files.reverse()
        self.op_type, self.op_args = op_type, op_args
        self.numbered_files = numbered_files[:num_realizations]

    def calc_write(self, bin_width=0.2, calc_upper_lower=True):
        op_type, op_args, numbered_files = self.op_type, self.op_args, self.numbered_files
        op = op_type(*op_args)  # starts with the last realization by default
        op.write(bin_width=bin_width)
        counts, op_corr = op.counts, op.op_corr * op.counts
        if calc_upper_lower:
            lower_counts, upper_counts, lower_op_corr, upper_op_corr = \
                op.lower.counts, op.upper.counts, op.lower.op_corr * op.lower.counts, op.upper.op_corr * op.upper.counts
        op_corr[counts == 0] = 0  # remove nans
        for i in numbered_files[1:]:  # from one before last forward
            op = op_type(*op_args, centers=np.loadtxt(os.path.join(self.sim_path, str(i))), spheres_ind=i)
            op.write(bin_width=bin_width, write_upper_lower=calc_upper_lower)
            counts += op.counts
            if op_type is not PositionalCorrelationFunction:
                I = np.where(op.counts > 0)
                op_corr[I] += op.op_corr[I] * op.counts[I]  # add psi where it is not nan
            if calc_upper_lower:
                lower_counts += op.lower.counts
                upper_counts += op.upper.counts
                if op_type is not PositionalCorrelationFunction:
                    I = np.where(op.lower.counts > 0)
                    lower_op_corr[I] += op.lower.op_corr[I] * op.lower.counts[I]  # add psi where it is not nan
                    I = np.where(op.upper.counts > 0)
                    upper_op_corr[I] += op.upper.op_corr[I] * op.upper.counts[I]  # add psi where it is not nan
        op.spheres_ind = str(numbered_files[-1]) + "-" + str(numbered_files[0])
        op.op_corr = op_corr / counts if op_type is not PositionalCorrelationFunction else counts / np.nanmean(
            counts[np.where(counts > 0)])
        op.counts = counts
        op.op_name = op.op_name + "_" + str(len(numbered_files)) + "_averaged"
        if calc_upper_lower:
            op.lower.op_corr = lower_op_corr / lower_counts if op_type is not PositionalCorrelationFunction else \
                lower_counts / np.nanmean(lower_counts[np.where(lower_counts > 0)])
            op.lower.counts = lower_counts
            op.lower.op_name = op.lower.op_name + "_" + str(len(numbered_files) + 1) + "_averaged"

            op.upper.op_corr = upper_op_corr / upper_counts if op_type is not PositionalCorrelationFunction else \
                upper_counts / np.nanmean(upper_counts[np.where(upper_counts > 0)])
            op.upper.counts = upper_counts
            op.upper.op_name = op.upper.op_name + "_" + str(len(numbered_files) + 1) + "_averaged"
        op.write(bin_width=bin_width, write_vec=False, write_upper_lower=calc_upper_lower)


class BurgerField(OrderParameter):
    def __init__(self, sim_path, a1, a2, psi_op, centers=None, spheres_ind=None, calc_upper_lower=True):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower=False)
        self.op_name = "burger_vectors"
        self.psi = psi_op.op_vec
        self.n_orientation = psi_op.m * psi_op.n
        self.a1, self.a2 = a1, a2
        if calc_upper_lower:
            upper_centers = [c for c in self.event_2d_cells.all_centers if
                             c[2] >= self.event_2d_cells.boundaries.edges[2] / 2]
            lower_centers = [c for c in self.event_2d_cells.all_centers if
                             c[2] < self.event_2d_cells.boundaries.edges[2] / 2]
            self.upper = BurgerField(sim_path, centers=upper_centers, spheres_ind=self.spheres_ind,
                                     calc_upper_lower=False,
                                     a1=a1 + a2, a2=a1 - a2, psi_op=psi_op.upper)
            self.lower = BurgerField(sim_path, centers=lower_centers, spheres_ind=self.spheres_ind,
                                     calc_upper_lower=False,
                                     a1=a1 + a2, a2=a1 - a2, psi_op=psi_op.lower)
            self.upper.op_name = "upper_burger_vectors"
            self.lower.op_name = "lower_burger_vectors"

    def calc_order_parameter(self, calc_upper_lower=True):
        perfect_lattice_vectors = np.array([n * self.a1 + m * self.a2 for n in range(-2, 2) for m in range(-2, 2)])
        disloc_burger, disloc_location = BurgerField.calc_burger_vector(
            self.event_2d_cells, self.psi, perfect_lattice_vectors, n_orientation=self.n_orientation)
        self.op_vec = np.concatenate((np.array(disloc_location).T, np.array(disloc_burger).T)).T  # x, y, bx, by field
        if calc_upper_lower:
            self.lower.calc_order_parameter(calc_upper_lower=False)
            self.upper.calc_order_parameter(calc_upper_lower=False)

    @staticmethod
    def calc_burger_vector(event_2d_cells, psi, perfect_lattice_vectors, n_orientation):
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
        wraped_centers, wraped_psi = BurgerField.wrap_with_boundaries(event_2d_cells, w=5, psi=psi)
        # all spheres within w distance from cyclic boundary will be mirrored
        tri = Delaunay(wraped_centers)
        dislocation_burger = []
        dislocation_location = []
        for i, simplex in enumerate(tri.simplices):
            rc = np.mean(tri.points[simplex], 0)
            if rc[0] < 0 or rc[0] > event_2d_cells.boundaries.edges[0] or rc[1] < 0 or rc[1] > \
                    event_2d_cells.boundaries.edges[1]:
                continue
            neighbor_points = []
            for neighbor_simplex_ind in tri.neighbors[i]:
                neighbor_points.append(
                    [tri.points[k] for k in tri.simplices[neighbor_simplex_ind] if k not in simplex][0])
            psi_avg = np.mean(wraped_psi[simplex])
            orientation = np.imag(np.log(psi_avg)) / n_orientation
            b_i = BurgerField.burger_calculation(tri.points[simplex], neighbor_points, perfect_lattice_vectors,
                                                 orientation)
            if np.linalg.norm(b_i) > 0:
                dislocation_location.append(rc)
                dislocation_burger.append(b_i)
        return dislocation_burger, dislocation_location

    @staticmethod
    def burger_calculation(simplex_points, neighbors_points, perfect_lattice_vectors, orientation):
        R = np.array([[np.cos(-orientation), -np.sin(-orientation)], [np.sin(-orientation), np.cos(-orientation)]])
        rotate = lambda ps: np.matmul(R, np.array(ps).T).T
        simplex_points = rotate(simplex_points)
        neighbors_points = rotate(neighbors_points)

        ts = []
        rc = np.mean(simplex_points, 0)
        for p in simplex_points:
            ts.append(np.arctan2(p[0] - rc[0], p[1] - rc[1]))
        I = np.argsort(ts)
        simplex_points = simplex_points[I]  # calculate burger circuit always anti-clockwise
        neighbors_points = neighbors_points[I]  # keep it connected to the k'th bond opposite to the k'th point
        which_L = lambda x_ab: np.argmin([np.linalg.norm(x_ab - L_) for L_ in perfect_lattice_vectors])
        L_ab = lambda x_ab: perfect_lattice_vectors[which_L(x_ab)]
        Ls = []
        for (a, b), k in zip([(0, 1), (1, 2), (2, 0)], [2, 0, 1]):
            x_c1, x_c2 = simplex_points[k], neighbors_points[k]
            x_a, x_b = simplex_points[a], simplex_points[b]
            L1 = L_ab(x_b - x_a)
            L2 = L_ab(x_b - x_c1) + L_ab(x_c1 - x_a)
            L3 = L_ab(x_b - x_c2) + L_ab(x_c2 - x_a)
            i_L = [which_L(L) for L in [L1, L2, L3]]
            Ls.append(perfect_lattice_vectors[np.argmax(np.bincount(i_L))])
        return np.sum(Ls, 0)

    @staticmethod
    def wrap_with_boundaries(event_2d_cells, w, psi):
        centers = np.array(event_2d_cells.all_centers)[:, :2]
        Lx, Ly = event_2d_cells.boundaries.edges[:2]
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

        psi1 = psi[np.logical_and(x - Lx > -w, y < w)]
        psi2 = psi[y < w]
        psi3 = psi[np.logical_and(x < w, y < w)]
        psi4 = psi[x - Lx > -w]
        psi5 = psi[:]
        psi6 = psi[x < w]
        psi7 = psi[np.logical_and(x - Lx > -w, y - Ly > -w)]
        psi8 = psi[y - Ly > -w]
        psi9 = psi[np.logical_and(x < w, y - Ly > -w)]

        wraped_centers = np.concatenate((sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9))
        wraped_psi = np.concatenate((psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8, psi9))
        return wraped_centers, wraped_psi

    def write(self, write_upper_lower=True):
        super().write(write_correlation=False, write_vec=True, write_upper_lower=False)
        if write_upper_lower:
            self.lower.write(write_upper_lower=False)
            self.upper.write(write_upper_lower=False)
            np.savetxt(os.path.join(os.path.join(self.sim_path, "OP"), "lower_" + str(self.spheres_ind) + ".txt"),
                       self.lower.event_2d_cells.all_centers)
            np.savetxt(os.path.join(os.path.join(self.sim_path, "OP"), "upper_" + str(self.spheres_ind) + ".txt"),
                       self.upper.event_2d_cells.all_centers)


def main():
    prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results2.0/"
    sim_path = os.path.join(prefix, sys.argv[1])
    calc_type = sys.argv[2]
    N = int(re.split('_h=', re.split('N=', sys.argv[1])[1])[0])
    correlation_couples = int(1e6)
    randomize = N ** 2 > correlation_couples
    op_dir = os.path.join(sim_path, "OP")
    if not os.path.exists(op_dir): os.mkdir(op_dir)
    log = os.path.join(op_dir, "log")
    sys.stdout = open(log, "a")
    print("\n\n\n-----------\nDate: " + str(date.today()) + "\nType: " + calc_type + "\nCorrelation couples: " + str(
        correlation_couples), file=sys.stdout)
    if calc_type == "psi23":
        psi23 = PsiMN(sim_path, 2, 3)
        psi23.calc_order_parameter()
        psi23.write(write_correlation=False, write_vec=True)
        psi23.correlation(low_memory=True, randomize=randomize, realizations=correlation_couples)
        psi23.write()
    if calc_type == "psi14":
        psi14 = PsiMN(sim_path, 1, 4)
        psi14.calc_order_parameter()
        psi14.write(write_correlation=False, write_vec=True)
        psi14.correlation(low_memory=True, randomize=randomize, realizations=correlation_couples)
        psi14.write()
    if calc_type == "psi16":
        psi16 = PsiMN(sim_path, 1, 6)
        psi16.calc_order_parameter()
        psi16.write(write_correlation=False, write_vec=True)
        psi16.correlation(low_memory=True, randomize=randomize, realizations=correlation_couples)
        psi16.write()
    if calc_type == "pos":
        psi23, psi14, psi16 = PsiMN(sim_path, 2, 3), PsiMN(sim_path, 1, 4), PsiMN(sim_path, 1, 6)
        psi23.calc_order_parameter(), psi14.calc_order_parameter(), psi16.calc_order_parameter()
        psis = [psi14.op_vec, psi23.op_vec, psi16.op_vec]
        correct_psi = psis[np.argmax(np.abs([np.sum(p) for p in psis]))]
        theta = np.angle(np.sum(correct_psi))
        pos = PositionalCorrelationFunction(sim_path, theta)
        pos.correlation(low_memory=True, randomize=randomize, realizations=correlation_couples)
        pos.write()
    if calc_type == "burger_square":
        psi14 = PsiMN(sim_path, 1, 4)
        psi14.calc_order_parameter()

        load = WriteOrLoad(output_dir=sim_path)
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = load.load_Input()
        a = np.sqrt(l_x * l_y / N)
        a1 = np.array([a, 0])
        a2 = np.array([0, a])

        burger = BurgerField(sim_path, a1, a2, psi14)
        burger.calc_order_parameter()
        burger.write()


if __name__ == "__main__":
    main()
