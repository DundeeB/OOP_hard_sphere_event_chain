import Structure
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
            u = Structure.CubeBoundaries.vertical_step_to_wall(wall, pos)
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
    FREE = 0  # Free path
    COLLISION = 1  # Path leads to collision with another sphere
    BC = 2  # Path reaches Boundary and needs to be handle using Boundary Condition


class Event:

    def __init__(self, sphere, dr, event_type):
        """
        A single event that will happen to sphere, such as collision with another sphere
        or arriving at the boundary of the simulation
        :param sphere: sphere which about to take a move
        :param dr: distance of the next move
        :param event_type: EventType enum: "Free path", "Collision", "Boundary Condition"
        """
        self.sphere = sphere
        self.dr = dr
        self.event_type = event_type



"""
    def step_size(self, positions, sphere_ind, v_hat, l):
        closest_wall, wall_type = self.wall_dist(positions[sphere_ind], v_hat, l)
        spheres_dists = [float('inf') for _ in range(len(positions))]
        for i in range(len(positions)):
            if i != sphere_ind:
                spheres_dists[i] = self.pair_dist(positions[sphere_ind], positions[i], l, v_hat)
        other_sphere = np.argmin(spheres_dists)
        dist_sphere = spheres_dists[other_sphere]
        if closest_wall < dist_sphere:  # it hits a wall
            return closest_wall, "wall", wall_type
        if closest_wall > dist_sphere:  # it hits another sphere
            return dist_sphere, "pair_collision", other_sphere
        return l, "Hits_nothing", np.nan
"""
