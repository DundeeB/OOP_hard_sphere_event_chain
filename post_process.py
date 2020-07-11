#!/Local/cmp/anaconda3/bin/python -u
import numpy as np
from SnapShot import *
from Structure import *
from EventChainActions import *
from sklearn.neighbors import *
import os
import sys


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
            self.upper = type(self)(sim_path, centers=upper_centers, spheres_ind=spheres_ind, calc_upper_lower=False,
                                    **kwargs)
            self.lower = type(self)(sim_path, centers=lower_centers, spheres_ind=spheres_ind, calc_upper_lower=False,
                                    **kwargs)

    def calc_order_parameter(self, calc_upper_lower=True):
        """to be override by child class"""
        pass

    def correlation(self, bin_width=0.2, calc_upper_lower=True, low_memory=False):
        if self.op_vec is None: self.calc_order_parameter()
        lx, ly = self.event_2d_cells.boundaries.edges[:2]
        l = np.sqrt(lx ** 2 + ly ** 2)
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
            for i, r in enumerate(self.event_2d_cells.all_centers):
                for j, r_ in enumerate(self.event_2d_cells.all_centers):
                    dr_vec = np.array(r) - r_
                    dx = np.min(np.abs([dr_vec[0], dr_vec[0] + lx, dr_vec[0] - lx]))
                    dy = np.min(np.abs([dr_vec[1], dr_vec[1] + ly, dr_vec[1] - ly]))
                    dr = np.sqrt(dx ** 2 + dy ** 2)
                    k = \
                        np.where(np.logical_and(centers - bin_width / 2 <= dr, centers + bin_width / 2 > dr))[0][0]
                    counts[k] += 1
                    phiphi_hist[k] += self.op_vec[i] * np.conjugate(self.op_vec[j])
        self.counts = counts
        self.op_corr = np.real(phiphi_hist) / counts + 1j * np.imag(phiphi_hist) / counts
        self.corr_centers = centers

        if calc_upper_lower:
            self.lower.correlation(bin_width, calc_upper_lower=False, low_memory=low_memory)
            self.upper.correlation(bin_width, calc_upper_lower=False, low_memory=low_memory)

    def calc_write(self, calc_correlation=True, bin_width=0.2, write_vec=False, calc_upper_lower=True):
        f = lambda a, b: os.path.join(a, b)
        g = lambda name, mat: np.savetxt(
            f(f(self.sim_path, "OP"), self.op_name + "_" + name + "_" + str(self.spheres_ind)) + ".txt", mat)
        if not os.path.exists(f(self.sim_path, "OP")): os.mkdir(f(self.sim_path, "OP"))
        if write_vec:
            if self.op_vec is None: self.calc_order_parameter()
            g("vec", self.op_vec)
        if calc_correlation:
            if self.op_corr is None: self.correlation(bin_width=bin_width)
            g("correlation", np.transpose([self.corr_centers, np.abs(self.op_corr), self.counts]))
        if calc_upper_lower:
            self.lower.calc_write(calc_correlation, bin_width, write_vec, calc_upper_lower=False)
            self.upper.calc_write(calc_correlation, bin_width, write_vec, calc_upper_lower=False)
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

    def calc_order_parameter(self, calc_upper_lower=True):
        self.op_vec, _ = PsiMN.psi_m_n(self.event_2d_cells, self.m, self.n)
        if calc_upper_lower:
            self.lower.calc_order_parameter(calc_upper_lower=False)
            self.upper.calc_order_parameter(calc_upper_lower=False)

    def calc_write(self, calc_correlation=True, bin_width=0.2, write_vec=True, calc_upper_lower=True):
        self.calc_order_parameter(calc_upper_lower=calc_upper_lower)
        super().calc_write(calc_correlation, bin_width, write_vec, calc_upper_lower)


class PositionalCorrelationFunction(OrderParameter):

    def __init__(self, sim_path, theta=0, rect_width=0.2, centers=None, spheres_ind=None, calc_upper_lower=True):
        super().__init__(sim_path, centers, spheres_ind, calc_upper_lower, theta=theta, rect_width=rect_width)
        self.theta = theta
        self.rect_width = rect_width
        self.op_name = "positional_theta=" + str(theta)
        if calc_upper_lower:
            self.upper.op_name = "upper_" + self.op_name
            self.lower.op_name = "lower_" + self.op_name

    def correlation(self, bin_width=0.2, calc_upper_lower=True, low_memory=False):
        theta, rect_width = self.theta, self.rect_width
        v_hat = np.transpose(np.matrix([np.cos(theta), np.sin(theta)]))
        lx, ly = self.event_2d_cells.boundaries.edges[:2]
        l = np.sqrt(lx ** 2 + ly ** 2)
        binds_edges = np.linspace(0, np.ceil(l / bin_width) * bin_width, int(np.ceil(l / bin_width)) + 1)
        self.corr_centers = binds_edges[:-1] + bin_width / 2
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
            I = np.where(dist_to_line <= rect_width/2)[0]
            pairs_dr = pairs_dr[I]
            J = np.where(pairs_dr * v_hat > 0)[0]
            pairs_dr = pairs_dr[J]
            rs = pairs_dr * v_hat
            self.counts, _ = np.histogram(rs, binds_edges)
        else:
            for r in self.event_2d_cells.all_centers:
                for r_ in self.event_2d_cells.all_centers:
                    dr = np.array(r) - r_
                    dxs = [dr[0], dr[0] + lx, dr[0] - lx]
                    dx = dxs[np.argmin(np.abs(dxs))]
                    dys = [dr[1], dr[1] + ly, dr[1] - ly]
                    dy = dys[np.argmin(np.abs(dys))]
                    dr = np.array([dx, dy])
                    dist_on_line = float(np.dot(dr, v_hat))
                    dist_vec = v_hat * dist_on_line - np.transpose(np.matrix(dr))
                    dist_to_line = np.linalg.norm(dist_vec)
                    if dist_to_line <= rect_width/2 and dist_on_line > 0:
                        k = np.where(
                            np.logical_and(binds_edges[:-1] <= dist_on_line, binds_edges[1:] > dist_on_line))[0][0]
                        self.counts[k] += 1
        self.op_corr = self.counts / np.nanmean(self.counts[np.where(self.counts > 0)])

        if calc_upper_lower:
            assert (self.upper is not None,
                    "Failed calculating upper positional correlation because it was not initialized")
            self.upper.correlation(bin_width=bin_width, calc_upper_lower=False)
            assert (self.upper is not None,
                    "Failed calculating lower positional correlation because it was not initialized")
            self.lower.correlation(bin_width=bin_width, calc_upper_lower=False)


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
        op.calc_write(bin_width=bin_width)
        counts, op_corr = op.counts, op.op_corr * op.counts
        if calc_upper_lower:
            lower_counts, upper_counts, lower_op_corr, upper_op_corr = \
                op.lower.counts, op.upper.counts, op.lower.op_corr * op.lower.counts, op.upper.op_corr * op.upper.counts
        op_corr[counts == 0] = 0  # remove nans
        for i in numbered_files[1:]:  # from one before last forward
            op = op_type(*op_args, centers=np.loadtxt(os.path.join(self.sim_path, str(i))), spheres_ind=i)
            op.calc_write(bin_width=bin_width, calc_upper_lower=calc_upper_lower)
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
        op.calc_write(bin_width=bin_width, write_vec=False, calc_upper_lower=calc_upper_lower)


def main():
    prefix = "/storage/ph_daniel/danielab/ECMC_simulation_results2.0/"
    sim_path = os.path.join(prefix, sys.argv[1])

    psi23 = PsiMN(sim_path, 2, 3)
    psi23.calc_order_parameter()
    psi23.correlation(low_memory=True)
    psi23.calc_write()

    psi14 = PsiMN(sim_path, 1, 4)
    psi14.calc_order_parameter()
    psi14.correlation(low_memory=True)
    psi14.calc_write()

    correct_psi = [psi14.op_vec, psi23.op_vec][np.argmax(np.abs([np.sum(psi14.op_vec), np.sum(psi23.op_vec)]))]
    theta = np.angle(np.sum(correct_psi))
    pos = PositionalCorrelationFunction(sim_path, theta)
    pos.calc_order_parameter()
    pos.correlation(low_memory=True)
    pos.calc_write()


if __name__ == "__main__":
    main()
