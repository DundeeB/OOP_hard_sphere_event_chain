import pylab, cv2, os, numpy as np, copy

from Structure import Sphere, CubeBoundaries, ArrayOfCells
from EventChainActions import Step


class View2D:

    def __init__(self, output_dir, boundaries, counter=None):
        """
        Create new View instance, saving pictures to directory output_dir and having simulation boundary
        boundaries
        :type output_dir: str
        :type boundaries: CubeBoundaries
        """
        self.output_dir = output_dir
        self.boundaries = boundaries
        self.counter = counter

    def plt_spheres(self, title, spheres):
        """

        :param title:
        :param spheres:
        :return:
        """
        pylab.gcf().set_size_inches(6, 6)
        pylab.cla()
        pylab.axis([0, self.boundaries.edges[0], 0, self.boundaries.edges[1]])
        for sphere in spheres:
            assert isinstance(sphere, Sphere)
            c = sphere.center
            circle = pylab.Circle((c[0], c[1]), radius=sphere.rad)
            pylab.gca().add_patch(circle)
        pylab.title(title)

    def dump_spheres(self, centers, file_name):
        output_dir = os.path.join(self.output_dir, file_name)
        np.savetxt(output_dir, centers)
        return

    def spheres_snapshot(self, title, spheres, img_name):
        self.plt_spheres(title, spheres)
        pylab.savefig(os.path.join(self.output_dir, img_name+".png"))

    @staticmethod
    def plt_step(sphere, vel, arrow_scale):
        x, y = sphere.center[0:2]
        circle = pylab.Circle((x, y),
                              radius=sphere.rad, fc='r')
        pylab.gca().add_patch(circle)

        (dx, dy) = vel[0:2]
        dx *= arrow_scale
        dy *= arrow_scale

        if dx == 0 and dy == 0: dx = 0.001
        pylab.arrow(x, y, dx, dy, fc="k", ec="k",
                    head_width=0.03 * np.linalg.norm(vel), head_length=0.03 * np.linalg.norm(vel))

    def step_snapshot(self, title, spheres, sphere_ind, img_name, vel, arrow_scale):
        self.plt_spheres(title, spheres)
        View2D.plt_step(spheres[sphere_ind], vel, arrow_scale)
        pylab.savefig(os.path.join(self.output_dir, img_name+".png"))

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
            for bound_vec in array_of_cells.boundaries.boundary_transformed_vectors():
                cloned_step.sphere.center = step.sphere.center + bound_vec
                View2D.plt_step(cloned_step.sphere, np.array(step.v_hat) * step.total_step, 0.1)
        for cell in array_of_cells.all_cells:
            x, y = cell.site
            rec = pylab.Rectangle((x, y), cell.edges[0], cell.edges[1], fill=False)
            pylab.gca().add_patch(rec)
        pylab.savefig(os.path.join(self.output_dir, img_name + ".png"))

    def save_video(self, video_name, fps):
        images = [img for img in os.listdir(self.output_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.output_dir, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(os.path.join(self.output_dir, video_name+".avi"),
                                0, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.output_dir, image)))

        cv2.destroyAllWindows()
        video.release()
