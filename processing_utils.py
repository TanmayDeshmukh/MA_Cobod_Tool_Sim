import numpy as np
from numpy import linalg as LA
import matplotlib

def filter_sample_points(samples: [[]], normals: [[]], adjacent_tool_pose_angle_threshold: float,
                         adjacent_vertex_angle_threshold: float, inter_ver_dist_thresh: float):
    ele_popped = 0
    popped_indices = []
    for i, point in enumerate(samples[1:-1]):

        point = np.array(point)
        # print('samples', samples, i)
        prev_point = np.array(samples[i - ele_popped])
        next_point = np.array(samples[i + 2 - ele_popped])
        a = point - prev_point
        b = next_point - point
        inter_normal_angle = np.arccos(np.clip(np.dot(normals[i - 1 - ele_popped], normals[i - ele_popped]), -1.0, 1.0))
        inter_vert_dist = LA.norm(a)
        inter_vert_angle = np.arccos(np.clip(np.dot(a, b) / (LA.norm(a) * LA.norm(b)), -1.0, 1.0))

        if (
                inter_normal_angle < adjacent_tool_pose_angle_threshold) or inter_vert_angle <= adjacent_vertex_angle_threshold:  # inter_vert_dist < inter_ver_dist_thresh  or
            # We dont't threshold inter-vert distances because elimination only depends on normal angles and how collinear the intermediate point is
            samples.pop(i + 1 - ele_popped)
            normals.pop((i + 1 - ele_popped))
            popped_indices.append(i + 1)
            ele_popped += 1
    return popped_indices


def angle_between_vectors(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def combine_subpaths(all_verts_this_section, all_normals, vert_dist_threshold, adjacent_tool_pose_angle_threshold):
    ele_popped = 0
    for i in range(len(all_verts_this_section) - 1):
        inter_vert_dist = LA.norm(np.array(all_verts_this_section[i - ele_popped][-1]) - np.array(
            all_verts_this_section[i + 1 - ele_popped][0]))
        inter_vert_angle = np.arccos(
            np.clip(np.dot(np.array(all_normals[i - ele_popped][-1]), np.array(all_normals[i + 1 - ele_popped][0])),
                    -1.0, 1.0))

        if inter_vert_dist < vert_dist_threshold and inter_vert_angle < adjacent_tool_pose_angle_threshold:
            all_verts_this_section[i + 1 - ele_popped].pop(0)  # remove first vertex in next group
            all_verts_this_section[i - ele_popped] += all_verts_this_section.pop(
                i + 1 - ele_popped)  # append next group to current group
            all_normals[i + 1 - ele_popped].pop(0)
            all_normals[i - ele_popped] += all_normals.pop(i + 1 - ele_popped)
            # print('popping in after', section_iter, len(all_verts_this_section))
            ele_popped += 1


def interpolate_tool_motion(all_tool_locations, all_tool_normals, sample_dist):
    all_tool_positions, tool_normals = [], []
    for position_pair, normal_pair in zip(all_tool_locations, all_tool_normals):
        point_1, point_2 = np.array(position_pair[0]), np.array(position_pair[1])
        movement_direction = (point_2 - point_1)
        movement_dist = LA.norm(movement_direction)
        movement_direction = movement_direction / movement_dist  # normalizing to get only direction vector
        continuous_tool_positions, continuous_tool_normals = [point_1], []
        n_samples = int(movement_dist / sample_dist) + 1
        while len(continuous_tool_positions) < n_samples:
            next_position = continuous_tool_positions[-1] + movement_direction * sample_dist
            continuous_tool_positions.append(next_position)
        n0, n1 = np.array(normal_pair[0]), np.array(normal_pair[1])
        omega = np.arccos(np.clip(np.dot(n0 / LA.norm(n0), n1 / LA.norm(n1)), -1.0,
                                  1.0))  # Clip so that we dont exceed -1.0, 1.0 due to float arithmatic errors
        so = np.sin(omega)
        if omega in [0.0, np.inf, np.nan]:
            # Two normals in the same direction, no need for slerp
            continuous_tool_normals = [-normal_pair[0]] * int(n_samples)
        else:
            # Spherical interpolation
            continuous_tool_normals = [-((np.sin((1.0 - t) * omega) / so) * n0 + (np.sin(t * omega) / so) * n1) for t in
                                       np.arange(0.0, 1.0, 1.0 / n_samples)]
        while len(continuous_tool_normals) > len(continuous_tool_positions):
            continuous_tool_normals.pop(-1)
        if len(continuous_tool_normals) < len(continuous_tool_positions):
            continuous_tool_normals.append(continuous_tool_normals[-1])

        all_tool_positions.append(continuous_tool_positions), tool_normals.append(continuous_tool_normals)
    return all_tool_positions, tool_normals


def affected_points_for_tool_positions(deposition_thickness, sample_tree, mesh, sample_face_indexes,
                                       sorted_intersection_locations, tool_positions, tool_normals, tool_major_axes, tool_minor_axes,
                                       gun_model, deposition_sim_time_resolution, scatter):
    # find points within sphere of radius of major axis
    for intersection_location, current_tool_position, current_tool_normal, current_tool_major_axis_vec, current_tool_minor_axis_vec in zip(
            sorted_intersection_locations, tool_positions, tool_normals, tool_major_axes, tool_minor_axes):
        query_ball_points = sample_tree.query_ball_point(intersection_location,
                                                         gun_model.b if gun_model.a <= gun_model.b else gun_model.a)
        # print('query_ball_points', len(query_ball_points), query_ball_points)

        for point_index in query_ball_points:
            point = sample_tree.data[point_index]

            tool_pos_to_point = point - current_tool_position
            tool_pos_to_point_dist = LA.norm(tool_pos_to_point)
            tool_pos_to_point /= tool_pos_to_point_dist

            angle_normal_to_point = angle_between_vectors(tool_pos_to_point, current_tool_normal)
            normal_dist_h_dash = np.cos(angle_normal_to_point) * tool_pos_to_point_dist
            rp = tool_pos_to_point * tool_pos_to_point_dist - current_tool_normal * normal_dist_h_dash
            angle_minor_axis_to_point = angle_between_vectors(rp / LA.norm(rp), current_tool_minor_axis_vec)
            d_rp = LA.norm(rp)

            x, y = d_rp * np.sin(angle_minor_axis_to_point), d_rp * np.cos(angle_minor_axis_to_point)
            if gun_model.check_point_validity(x, y):
                # Estimate deposition thickness for this point
                surface_normal = mesh.face_normals[sample_face_indexes[point_index]]
                deposition_at_h = gun_model.deposition_intensity(x, y)
                multiplier = ((gun_model.h / tool_pos_to_point_dist) ** 2) * np.dot(surface_normal,
                                                                                    tool_pos_to_point) / (
                                 np.dot(tool_pos_to_point, -current_tool_normal))
                # multiplier = ((gun_model.h / tool_pos_to_point_dist))
                # multiplier = 1
                deposition_thickness[point_index] += multiplier * deposition_at_h * deposition_sim_time_resolution

        n = matplotlib.colors.Normalize(vmin=min(deposition_thickness),
                                        vmax=max(deposition_thickness))
        m = matplotlib.cm.ScalarMappable(norm=n, cmap='YlOrBr_r')
        scatter.set_color(m.to_rgba(deposition_thickness))
        scatter._facecolor3d = scatter.get_facecolor()
        scatter._edgecolor3d = scatter.get_edgecolor()

        matplotlib.pyplot.draw()
        matplotlib.pyplot.pause(0.0000001)