import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.io as sio
from EventChainActions import Step
from Structure import *


class WriteOrLoad:

    def __init__(self, output_dir, boundaries=[], counter=None):
        """
        Create new View instance, saving pictures to directory output_dir and having simulation boundary
        boundaries
        :type output_dir: str
        :type boundaries: list
        """
        self.output_dir = output_dir
        self.boundaries = boundaries
        self.counter = counter

    def plt_spheres(self, title, spheres, h=0):
        """
        :param title: of figure
        :param spheres: list of obj type Sphere
        :param h: height of 3D simulation
        """
        plt.gcf().set_size_inches(6, 6)
        plt.cla()
        plt.axis([0, self.boundaries[0], 0, self.boundaries[1]])
        for sphere in spheres:
            assert isinstance(sphere, Sphere)
            c = sphere.center
            if len(c) == 3 and c[-1] > 2 * sphere.rad*(h + 1) / 2:
                color = 'r'
            else:
                color = 'b'
            circle = plt.Circle((c[0], c[1]), radius=sphere.rad, color=color, alpha=.2)
            plt.gca().add_patch(circle)
        plt.title(title)

    def dump_spheres(self, centers, file_name):
        output_dir = os.path.join(self.output_dir, file_name)
        np.savetxt(output_dir, centers)
        return

    def spheres_snapshot(self, title, spheres, img_name):
        self.plt_spheres(title, spheres)
        plt.savefig(os.path.join(self.output_dir, img_name + ".png"))

    @staticmethod
    def plt_step(sphere, vel, arrow_scale):
        x, y = sphere.center[0:2]
        circle = plt.Circle((x, y),
                            radius=sphere.rad, fc='r')
        plt.gca().add_patch(circle)

        (dx, dy) = vel[0:2]
        dx *= arrow_scale
        dy *= arrow_scale

        if dx == 0 and dy == 0: dx = 0.001
        plt.arrow(x, y, dx, dy, fc="k", ec="k",
                  head_width=0.03 * np.linalg.norm(vel), head_length=0.03 * np.linalg.norm(vel))

    def step_snapshot(self, title, spheres, sphere_ind, img_name, vel, arrow_scale):
        self.plt_spheres(title, spheres)
        WriteOrLoad.plt_step(spheres[sphere_ind], vel, arrow_scale)
        plt.savefig(os.path.join(self.output_dir, img_name + ".png"))

    def array_of_cells_snapshot(self, title, array_of_cells, img_name, step=None):
        """
        :param title: title of the figure
        :type array_of_cells: ArrayOfCells
        :param img_name: file name
        :type step: Step
        """
        spheres = array_of_cells.cushioning_array_for_boundary_cond().all_spheres
        self.plt_spheres(title, spheres)
        if step is not None:
            cloned_step = copy.deepcopy(step)
            cloned_step.sphere.center = step.sphere.center
            v_hat = np.array([0, 0, 0])
            v_hat[step.direction.dim] = step.direction.sgn
            WriteOrLoad.plt_step(cloned_step.sphere, v_hat * step.total_step, 0.1)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                c = self.cells[i][j]
                x, y = c.site
                rec = plt.Rectangle((x, y), c.edges[0], c.edges[1], fill=False)
                plt.gca().add_patch(rec)
        plt.savefig(os.path.join(self.output_dir, img_name + ".png"))

    def save_video(self, video_name, fps):
        images = [img for img in os.listdir(self.output_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.output_dir, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(os.path.join(self.output_dir, video_name + ".avi"),
                                0, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.output_dir, image)))

        cv2.destroyAllWindows()
        video.release()

    def save_Input(self, rad, rho_H, edge, n_row, n_col):
        file_name = os.path.join(self.output_dir, 'Input_parameters_from_python.mat')
        l_x, l_y, l_z = self.boundaries
        sio.savemat(file_name, {'rad': float(rad), 'Lx': l_x, 'Ly': l_y,
                                'H': float(l_z), 'rho_H': rho_H, 'edge': edge, 'n_row': n_row, 'n_col': n_col})

    def realizations(self):
        if not os.path.exists(self.output_dir):
            raise ValueError("No folder found: " + self.output_dir)
        files = os.listdir(self.output_dir)
        numbered_files = [int(f) for f in files if re.findall("^\d+$", f)]
        if len(numbered_files) > 0:
            numbered_files.sort(reverse=True)
            return numbered_files
        else:
            Warning("No realizations found")
            if os.path.exists(os.path.join(self.output_dir, 'Initial Conditions')):
                return [0]
            else:
                raise ValueError("No file center found at folder: " + self.output_dir)

    def last_spheres(self, k=0):
        file_ind = self.realizations()[k]
        sp_name = str(file_ind) if file_ind > 0 else 'Initial Conditions'
        return np.loadtxt(os.path.join(self.output_dir, sp_name)), file_ind

    def load_Input(self):
        file_name = os.path.join(self.output_dir, 'Input_parameters_from_python.mat')
        dictionary = sio.loadmat(file_name)
        l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col = \
            dictionary['Lx'][0][0], dictionary['Ly'][0][0], dictionary['H'][0][0], dictionary['rad'][0][0], \
            dictionary['rho_H'][0][0], dictionary['edge'][0][0], dictionary['n_row'][0][0], dictionary['n_col'][0][0]
        return l_x, l_y, l_z, rad, rho_H, edge, n_row, n_col
