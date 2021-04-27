import numpy as np
from numpy import linalg as LA
import itertools


class TrajectoryGenerator:

    def __init__(self, mesh, gun_model):

        self.mesh = mesh
        self.gun_model = gun_model
        self.standoff_dist = gun_model.h
        self.direction_flag = False
        self.extend_trajectory_outside = False
        self.vert_dist_threshold = 0.05  # m
        self.adjacent_tool_pose_angle_threshold = np.radians(1.0)
        self.adjacent_vertex_angle_threshold = np.radians(1.0)

    def generate_trajectory(self, d3sections):

        all_tool_locations = []
        all_tool_normals = []
        section_end_vert_pairs = []

        for section_iter, section_path_group in enumerate(d3sections):
            face_indices = section_path_group.metadata['face_index']

            all_verts_this_section = []
            tool_normals_this_section = []
            self.direction_flag = not self.direction_flag
            face_count_up = 0
            for subpath_iter, subpath in enumerate(section_path_group.entities):
                subpath_tool_positions = []
                subpath_tool_normals = []
                for line_segment_index in range(len(subpath.points) - 1):
                    this_face_normal = self.mesh.face_normals[face_indices[face_count_up]]
                    face_count_up += 1

                    vert1_index, vert2_index = subpath.points[line_segment_index], subpath.points[line_segment_index + 1]
                    vert1, vert2 = section_path_group.vertices[vert1_index], section_path_group.vertices[vert2_index]

                    new_ver1 = this_face_normal * self.standoff_dist + vert1
                    new_ver2 = this_face_normal * self.standoff_dist + vert2

                    subpath_tool_positions.append([x for x in new_ver1])
                    subpath_tool_positions.append([x for x in new_ver2])
                    subpath_tool_normals.append(-this_face_normal)
                    subpath_tool_normals.append(-this_face_normal)

                    # plot_normals(visualizer.axs_init, vertices=[np.array(new_ver1)], directions=[np.array(this_face_normal)])

                # check first 2 z values and correct the subpaths' direction
                # plot_path(visualizer.axs_init, np.array(subpath_tool_positions))

                if (subpath_tool_positions[0][2] > subpath_tool_positions[1][2]) ^ self.direction_flag:
                    subpath_tool_positions.reverse()
                    subpath_tool_normals.reverse()

                all_verts_this_section.append(subpath_tool_positions)
                tool_normals_this_section.append(subpath_tool_normals)

                subpath_tool_positions = np.array(subpath_tool_positions)

            # Correct the order of subpaths first (bubble sorting)
            for i in range(len(all_verts_this_section)):
                for j in range(len(all_verts_this_section) - 1):
                    if (all_verts_this_section[j][0][2] > all_verts_this_section[j + 1][0][2]) ^ self.direction_flag:
                        all_verts_this_section[j], all_verts_this_section[j + 1] = all_verts_this_section[j + 1], \
                                                                                   all_verts_this_section[j]
                        tool_normals_this_section[j], tool_normals_this_section[j + 1] = tool_normals_this_section[j + 1], \
                                                                                         tool_normals_this_section[j]

            # Combine sub-paths if endpoints are close enough. This must be done before removing unnecessary intermediate points
            # because the end points of the sub paths themselves might be unnecessary
            self.combine_subpaths(all_verts_this_section, tool_normals_this_section, self.vert_dist_threshold, self.adjacent_tool_pose_angle_threshold)

            # Remove unnecessary intermediate points
            for vert_group, norms in zip(all_verts_this_section, tool_normals_this_section):
                self.filter_sample_points(vert_group, norms, adjacent_tool_pose_angle_threshold=self.adjacent_tool_pose_angle_threshold,
                                     adjacent_vertex_angle_threshold=self.adjacent_vertex_angle_threshold,
                                     inter_ver_dist_thresh=self.vert_dist_threshold)
                vert_group = np.array(vert_group)

            # Extend tool travel outward for better coverage
            if self.extend_trajectory_outside:
                for subpath_tool_positions in all_verts_this_section:
                    start_direction = np.array(subpath_tool_positions[0]) - np.array(subpath_tool_positions[1])
                    start_direction /= LA.norm(start_direction)
                    start_direction *= self.gun_model.b
                    end_direction = np.array(subpath_tool_positions[-1]) - np.array(subpath_tool_positions[-2])
                    end_direction /= LA.norm(end_direction)
                    end_direction *= self.gun_model.b

                    subpath_tool_positions[0] = [sum(x) for x in zip(subpath_tool_positions[0], start_direction.tolist())]
                    subpath_tool_positions[-1] = [sum(x) for x in zip(subpath_tool_positions[-1], end_direction.tolist())]

            all_tool_normals += tool_normals_this_section
            tool_normals_this_section = -np.array(list(itertools.chain.from_iterable(tool_normals_this_section)))

            # Visualization of activated(g) and2    `deactivated(k) tool travel within this section cut
            all_tool_locations += all_verts_this_section
            all_verts_this_section = list(itertools.chain.from_iterable(all_verts_this_section))
            all_verts_this_section = np.array(all_verts_this_section)
            section_end_vert_pairs += [all_verts_this_section[-1]] if section_iter == 0 else [all_verts_this_section[0],
                                                                                              all_verts_this_section[-1]]
        return all_tool_locations, all_tool_normals, section_end_vert_pairs

    def filter_sample_points(self, samples: [[]], normals: [[]], adjacent_tool_pose_angle_threshold: float,
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
            inter_normal_angle = np.arccos(
                np.clip(np.dot(normals[i - 1 - ele_popped], normals[i - ele_popped]), -1.0, 1.0))
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

    def combine_subpaths(self, tool_positions: [[]], tool_normals: [[]], vert_dist_threshold: float,
                         adjacent_tool_normal_angle_threshold: float):
        ele_popped = 0
        print('tool_positions combine_subpaths', tool_positions)
        for i in range(len(tool_positions) - 1):
            inter_vert_dist = LA.norm(np.array(tool_positions[i - ele_popped][-1]) - np.array(
                tool_positions[i + 1 - ele_popped][0]))
            inter_vert_angle = np.arccos(
                np.clip(
                    np.dot(np.array(tool_normals[i - ele_popped][-1]), np.array(tool_normals[i + 1 - ele_popped][0])),
                    -1.0, 1.0))

            if inter_vert_dist < vert_dist_threshold and inter_vert_angle < adjacent_tool_normal_angle_threshold:
                tool_positions[i + 1 - ele_popped].pop(0)  # remove first vertex in next group
                tool_positions[i - ele_popped] += tool_positions.pop(
                    i + 1 - ele_popped)  # append next group to current group
                tool_normals[i + 1 - ele_popped].pop(0)
                tool_normals[i - ele_popped] += tool_normals.pop(i + 1 - ele_popped)
                # print('popping in after', section_iter, len(tool_positions))
                ele_popped += 1
