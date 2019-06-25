from Structure import *
import numpy as np
from enum import Enum

epsilon = 1e-4


class Metric:

    @staticmethod
    def dist_to_boundary(sphere, v_hat, l, boundaries):
        """
        Figures out the distance for the next boundary
        :param sphere: sphere to be moved
        :param v_hat: direction of step
        :param l: magnitude of the step to be carried out
        :param boundaries: of the simulation
        :return: the minimal distance to the wall, and the wall.
        If there is no wall in a distance l, dist is inf and wall=[]
        """
        v_hat = np.array(v_hat) / np.linalg.norm(v_hat)
        pos = np.array(sphere.center)
        r = sphere.rad
        min_dist_to_wall = float('inf')
        closest_wall = []
        for wall in boundaries.get_walls():
            u = CubeBoundaries.vertical_step_to_wall(wall, pos)
            u = u - r*u/np.linalg.norm(u) #shift the wall closer by r
            if np.dot(u, v_hat) < 0:
                continue
            v = np.dot(u, u)/np.dot(u, v_hat)
            if v < min_dist_to_wall and v <= l:
                min_dist_to_wall = v
                closest_wall = wall
        return min_dist_to_wall, closest_wall

    @staticmethod
    def dist_to_collision(sphere1, sphere2, l, v_hat):
        """
        Distance sphere1 would need to go in v_hat direction in order to collide with sphere2
        :param sphere1: sphere about to move
        :param sphere2: potential for collision
        :param l: maximal step size. If dist for collision > l then dist_to_collision-->infty
        :param v_hat: direction in which sphere1 is to move
        :return: distance for collision if the move is allowed, infty if move can not lead to collision
        """
        pos_a = sphere1.center
        pos_b = sphere2.center
        d = sphere1.rad + sphere2.rad
        v_hat = np.array(v_hat)/np.linalg.norm(v_hat)
        dx = np.array(pos_b) - np.array(pos_a)
        dx_dot_v = np.dot(dx, v_hat)
        if dx_dot_v <= 0:
            return float('inf')
        if np.linalg.norm(dx) - d <= epsilon:
            return 0
        discriminant = dx_dot_v ** 2 + d ** 2 - np.linalg.norm(dx) ** 2
        if discriminant > 0:
            dist: float = dx_dot_v - np.sqrt(discriminant)
            if dist <= l and dist >= 0: return dist
        return float('inf')


class EventType(Enum):
    FREE = "FreeStep"  # Free path
    COLLISION = "SphereSphereCollision"  # Path leads to collision with another sphere
    BC = "BoundaryCondition"  # Path reaches Boundary and needs to be handle using Boundary Condition


class Event:

    def __init__(self, step, event_type, other_sphere, wall):
        """
        A single event that will happen to sphere, such as collision with another sphere
        or arriving at the boundary of the simulation
        :param step: step to be perform before event
        :param event_type: EventType enum: "Free path", "Collision", "Boundary Condition"
        :param other_sphere: if event is collision, this is the sphere it will collide
        :param wall: if event is boundary condition, this is the wall it is going to reach
        """
        self.step = step
        self.event_type = event_type
        self.other_sphere = other_sphere
        self.wall = wall


class Step:

    def __init__(self, sphere, total_step, current_step, v_hat):
        """
        The characteristic of the step to be perform
        :param sphere: sphere which about to move
        :param total_step: total step left for the current move of spheres
        :param current_step: step sphere is about to perform
        :param v_hat: direction of the step
        """
        self.sphere = sphere
        self.total_step = total_step
        self.current_step = current_step
        self.v_hat = v_hat

    def perform_step(self):
        self.sphere.perform_step(self)
        self.total_step = self.total_step - self.current_step


class EventChainActions:

    def __init__(self, boundaries):
        self.boundaries = boundaries

    def next_event(self, sphere, other_spheres, total_step, v_hat):
        """
        Returns the next Event object to be handle, such as from the even get the step, perform the step and decide the
        next event
        :param sphere: sphere wishing to perform the step
        :param other_spheres: other spheres which sphere might collide
        :param total_step: total step to be perform at current iteration by spheres
        :param v_hat: direction of step
        :return: Step object containing the needed information such as step size or step type (wall free or boundary)
        """
        min_dist_to_wall, closest_wall = Metric.dist_to_boundary(sphere, v_hat, total_step, self.boundaries)
        closest_sphere = []
        closest_sphere_dist = float('inf')
        for other_sphere in other_spheres:
            sphere_dist = Metric.dist_to_collision(sphere, other_sphere, total_step, v_hat)
            if sphere_dist < closest_sphere_dist:
                closest_sphere_dist = sphere_dist
                closest_sphere = other_sphere

        # it hits a wall
        if min_dist_to_wall < closest_sphere_dist:
            step = Step(sphere, total_step, min_dist_to_wall, v_hat)
            return Event(step, EventType.BC, [], closest_wall)

        # it hits another sphere
        if min_dist_to_wall > closest_sphere_dist:
            step = Step(sphere, total_step, closest_sphere_dist, v_hat)
            return Event(step, EventType.COLLISION, closest_sphere, [])

        # it hits nothing, both min_dist_to_wall and closest_sphere_dist are inf
        step = Step(sphere, total_step, total_step, v_hat)
        return Event(step, EventType.FREE, [], [])
