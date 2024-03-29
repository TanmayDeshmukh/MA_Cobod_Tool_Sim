import numpy as np
from numpy import linalg as LA
import matplotlib
import time

import viz_utils
from viz_utils import *


def angle_between_vectors(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def interpolate_tool_motion(all_tool_locations: [], all_tool_normals: [], sample_dist: float):
    all_tool_positions, tool_normals = [], []
    for segment_tool_positions, segment_tool_normals in zip(all_tool_locations, all_tool_normals):
        for curr_index in range(len(segment_tool_positions)-1):
            position_pair = segment_tool_positions[curr_index], segment_tool_positions[curr_index+1]
            normal_pair = segment_tool_normals[curr_index], segment_tool_normals[curr_index + 1]
            point_1, point_2 = np.array(position_pair[0]), np.array(position_pair[1])
            movement_direction = (point_2 - point_1)
            movement_dist = LA.norm(movement_direction)
            movement_direction = movement_direction / movement_dist  # normalizing to get only direction vector
            continuous_tool_positions, continuous_tool_normals = [point_1], []
            # TODO: Need better interpolation (Maybe there's change in orientation without change in position)
            n_samples = 2
            if movement_dist>=sample_dist:
                n_samples = int(movement_dist / sample_dist) + 1 # to include last point

            while len(continuous_tool_positions) < n_samples:
                next_position = continuous_tool_positions[-1] + movement_direction * sample_dist
                continuous_tool_positions.append(next_position)
            n0, n1 = np.array(normal_pair[0]), np.array(normal_pair[1])
            omega = np.arccos(np.clip(np.dot(n0 / LA.norm(n0), n1 / LA.norm(n1)), -1.0,
                                      1.0))  # Clip so that we dont exceed -1.0, 1.0 due to floating arithmatic errors
            so = np.sin(omega)
            if omega in [0.0, np.inf, np.nan]:
                # Two normals in the same direction, no need for slerp
                continuous_tool_normals = [normal_pair[0]] * int(n_samples)
            else:
                # Spherical interpolation
                continuous_tool_normals = [((np.sin((1.0 - t) * omega) / so) * n0 + (np.sin(t * omega) / so) * n1) for t in
                                           np.arange(0.0, 1.0, 1.0 / n_samples)]
            while len(continuous_tool_normals) > len(continuous_tool_positions):
                continuous_tool_normals.pop(-1)
            if len(continuous_tool_normals) < len(continuous_tool_positions):
                continuous_tool_normals.append(continuous_tool_normals[-1])

            all_tool_positions.append(continuous_tool_positions), tool_normals.append(continuous_tool_normals)
    return all_tool_positions, tool_normals


def closest_reachable_position(tool_position: [], x_lims: [], y_lims: [], z_lims: []) -> np.ndarray:
    tool_position = np.array(tool_position)
    new_pos = np.array([np.clip(tool_position[0], x_lims[0], x_lims[1]),
                        np.clip(tool_position[1], y_lims[0], y_lims[1]),
                        np.clip(tool_position[2], z_lims[0], z_lims[1])])
    return new_pos


def can_tool_reach_position(tool_position: [], tool_limits) -> (bool, np.ndarray):
    new_pos = closest_reachable_position(tool_position, *tool_limits)
    if (np.array(tool_position) == new_pos).all():
        return True, new_pos
    else:
        return False, new_pos



def limit_tool_position(desired_tool_position, surface_projected_position, desired_tool_normal, tool_limits ) -> (np.ndarray, np.ndarray):

    reachable, new_position = can_tool_reach_position(desired_tool_position, tool_limits)
    new_normal = desired_tool_normal
    if not reachable:
        new_normal = surface_projected_position-new_position
        new_normal /= LA.norm(new_normal)

    return new_position, new_normal


def get_intersection_point(tool_position, tool_normal, mesh, ray_mesh_intersector, point_sample_tree):

    intersection_location, index_ray, intersection_index_tri = ray_mesh_intersector.intersects_location(
        [tool_position],
        [tool_normal])
    surface_normal = []
    if len(intersection_location) > 0:
        intersection_location = intersection_location[0]
        surface_normal = mesh.face_normals[intersection_index_tri[0]]
    else:
        # no solution found
        intersection_location = get_virtual_intersection_point(tool_position, tool_normal, point_sample_tree)
        surface_normal = -tool_normal
    return intersection_location, surface_normal


def get_virtual_intersection_point(tool_position, tool_normal, point_sample_tree):
    closest_points = point_sample_tree.data[point_sample_tree.query(tool_position, k=10)[1]] # returns [(distances), (indexes)]

    shortest_perp_dist = 1000
    vir_int_pt = []

    for point in closest_points:
        tool_pos_to_point = point - tool_position
        tool_pos_to_point_dist = LA.norm(tool_pos_to_point)

        normal_dist_h_dash = LA.norm(np.dot(tool_pos_to_point, tool_normal))
        rp = tool_pos_to_point - tool_normal * normal_dist_h_dash

        rp_dist = LA.norm(rp)
        if rp_dist< shortest_perp_dist:
            shortest_perp_dist = rp_dist
            vir_int_pt = tool_position+tool_normal*normal_dist_h_dash
    viz_utils.visualizer.axs_init.scatter(vir_int_pt[0],vir_int_pt[1], vir_int_pt[2], s=30, c='r')
    return vir_int_pt


def surface_scaling(expected_h, actual_h, surface_normal: [], tool_pos_to_point_vec: [], tool_normal: []) -> float:
    multiplier = ((expected_h / actual_h) ** 2) * np.dot(tool_pos_to_point_vec, surface_normal) / (
                     np.dot(tool_pos_to_point_vec, -tool_normal))
    return multiplier

