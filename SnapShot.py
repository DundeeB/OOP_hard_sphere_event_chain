import pylab
import cv2
import os
import numpy as np

from Structure import Sphere, CubeBoundaries, ArrayOfCells


class View2D:

    def __init__(self, output_dir, boundaries):
        """
        Create new View instance, saving pictures to directory output_dir and having simulation boundary
        boundaries
        :type output_dir: str
        :type boundaries: CubeBoundaries
        """
        self.output_dir = output_dir
        self.boundaries = boundaries

    def plt_spheres(self, title, spheres):
        pylab.gcf().set_size_inches(6, 6)
        pylab.cla()
        pylab.axis([0, self.boundaries.edges[0], 0, self.boundaries.edges[1]])
        for sphere in spheres:
            assert isinstance(sphere, Sphere)
            x, y = sphere.center
            circle = pylab.Circle((x, y), radius=sphere.rad)
            pylab.gca().add_patch(circle)
        pylab.title(title)

    def spheres_snapshot(self, title, spheres, img_name):
        self.plt_spheres(title, spheres)
        pylab.savefig(os.path.join(self.output_dir, img_name+".png"))

    def step_snapshot(self, title, spheres, sphere_ind, img_name, vel, arrow_scale):
        self.plt_spheres(title, spheres)

        x, y = spheres[sphere_ind].center
        circle = pylab.Circle((x, y),
                              radius=spheres[sphere_ind].rad, fc='r')
        pylab.gca().add_patch(circle)

        (dx, dy) = vel
        dx *= arrow_scale
        dy *= arrow_scale
        pylab.arrow(x, y, dx, dy, fc="k", ec="k",
                    head_width=0.1*np.linalg.norm(vel), head_length=0.1*np.linalg.norm(vel))

        pylab.savefig(os.path.join(self.output_dir, img_name+".png"))

    def array_of_cells_snapshot(self, title, array_of_cells, img_name):
        """
        :type array_of_cells: ArrayOfCells
        """
        spheres = array_of_cells.all_spheres()
        self.plt_spheres(title, spheres)
        for cell in array_of_cells.all_cells():
            x, y = cell.site
            rec = pylab.Rectangle((x, y), cell.edges[0], cell.edges[1], fill=False)
            pylab.gca().add_patch(rec)
        pylab.savefig(os.path.join(self.output_dir, img_name + ".png"))

    def save_video(self, video_name):
        images = [img for img in os.listdir(self.output_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.output_dir, images[0]))
        height, width, layers = frame.shape
        fps = 10
        video = cv2.VideoWriter(os.path.join(self.output_dir, video_name+".avi"),
                                0, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.output_dir, image)))

        cv2.destroyAllWindows()
        video.release()
