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
        print('intersection_location', intersection_location, 'surface_normal', surface_normal)
    return intersection_location, surface_normal


def get_virtual_intersection_point(tool_position, tool_normal, point_sample_tree):
    closest_points = point_sample_tree.data[point_sample_tree.query(tool_position, k=10)[1]] # returns [(distances), (indexes)]

    shortest_perp_dist = 1000
    clostest_point = []

    for point in closest_points:
        tool_pos_to_point = point - tool_position
        tool_pos_to_point_dist = LA.norm(tool_pos_to_point)

        normal_dist_h_dash = np.dot(tool_normal, tool_pos_to_point)
        rp = tool_pos_to_point - tool_normal * normal_dist_h_dash

        rp_dist = LA.norm(rp)
        if rp_dist< shortest_perp_dist:
            shortest_perp_dist = rp_dist
            clostest_point = point-rp
    viz_utils.visualizer.final_path_ax.scatter(clostest_point[0],clostest_point[1], clostest_point[2], s=30, c='r')
    return clostest_point

def affected_points_for_tool_position(deposition_thickness, sample_tree, mesh,
                                       surface_normal, intersection_location,
                                       tool_position, tool_normal,
                                       tool_major_axis_vec, tool_minor_axis_vec,
                                       gun_model, deposition_sim_time_resolution, scatter):

    # Take the larger axis and double it = search radius
    query_ball_points = sample_tree.query_ball_point(intersection_location,
                                                     (gun_model.b if gun_model.a <= gun_model.b else gun_model.a) * 2.0)
    k = 0
    for point_index in query_ball_points:
        point = sample_tree.data[point_index]
        # print(k, end=' ')
        k += 1

        tool_pos_to_point = point - tool_position
        tool_pos_to_point_dist = LA.norm(tool_pos_to_point)

        normal_dist_h_dash = np.dot(tool_normal, tool_pos_to_point)
        rp = tool_pos_to_point - tool_normal * normal_dist_h_dash

        """
        # Debugging
        viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [tool_position], [tool_normal], color='g', lw=1)
        viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [tool_position], [rp/LA.norm(rp)], color='b', lw=1, norm_length=LA.norm(rp))
        viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [tool_position], [tool_pos_to_point/LA.norm(tool_pos_to_point)], color='r', lw=1, norm_length = LA.norm(tool_pos_to_point))
        """
        x, y = np.dot(rp, tool_major_axis_vec), np.dot(rp, tool_minor_axis_vec)

        if gun_model.check_point_validity(x, y):

            # Estimate deposition thickness for this point
            normal_dist_actual = LA.norm(intersection_location-tool_position)
            gun_model.set_h(normal_dist_h_dash)
            deposition_at_h_dash = gun_model.deposition_intensity(x, y)
            # multiplier = 1/((normal_dist_actual/tool_pos_to_point_dist)**2)*np.dot(tool_pos_to_point/tool_pos_to_point_dist, -surface_normal)/np.dot(tool_pos_to_point/tool_pos_to_point_dist, tool_normal)**3
            multiplier = ((normal_dist_actual/normal_dist_h_dash)**2)*np.dot(tool_pos_to_point/LA.norm(tool_pos_to_point), -surface_normal)
            # multiplier = 1
            # print('yes', multiplier)
            deposition_thickness[point_index] += multiplier * deposition_at_h_dash * deposition_sim_time_resolution
    # print('\nDrawing', max(deposition_thickness))
    n = matplotlib.colors.Normalize(vmin=min(deposition_thickness),
                                    vmax=max(deposition_thickness))
    m = matplotlib.cm.ScalarMappable(norm=n, cmap='YlOrBr_r')
    scatter.set_color(m.to_rgba(deposition_thickness))
    scatter._facecolor3d = scatter.get_facecolor()
    scatter._edgecolor3d = scatter.get_edgecolor()
    # print('\ndone')


def surface_scaling(expected_h, actual_h, surface_normal: [], tool_pos_to_point_vec: [], tool_normal: []) -> float:
    multiplier = ((expected_h / actual_h) ** 2) * np.dot(surface_normal,
                                                                        tool_pos_to_point_vec) / (
                     np.dot(tool_pos_to_point_vec, -tool_normal))
    return multiplier

