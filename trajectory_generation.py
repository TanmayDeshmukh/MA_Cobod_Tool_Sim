import numpy as np
from numpy import linalg as LA
import itertools
import viz_utils

class TrajectoryGenerator:

    def __init__(self, mesh, gun_model, standoff_dist):

        self.mesh = mesh
        self.gun_model = gun_model
        self.standoff_dist = standoff_dist
        self.direction_flag = False
        self.extend_trajectory_outside = False
        self.vert_dist_threshold =  0.005  # m
        self.adjacent_tool_pose_angle_threshold = np.radians(10.0)
        self.adjacent_vertex_angle_threshold = np.radians(170.0)

    def generate_trajectory(self, d3sections):

        all_tool_locations = []
        all_tool_normals = []
        section_end_vert_pairs = []

        for section_iter, section_path_group in enumerate(d3sections):
            face_indices = section_path_group.metadata['face_index']
            print('\n\nface_ind', face_indices)
            print('section_path_group.entities', section_path_group.entities.shape)
            all_verts_this_section = []
            tool_normals_this_section = []
            self.direction_flag = not self.direction_flag
            face_count_up = 0
            for subpath_iter, subpath in enumerate(section_path_group.entities):
                subpath_tool_positions = []
                subpath_tool_normals = []
                print(subpath.points, 'f', face_count_up, end=' : ')
                all_verts = section_path_group.vertices[subpath.points]
                viz_utils.visualizer.axs_slice.scatter(all_verts[:, 0],all_verts[:, 1],all_verts[:, 2], s=20, c='r')
                #viz_utils.plot_path(viz_utils.visualizer.axs_slice, vertices=all_verts,
                #                    color='g', lw=1.5, hw=0.3)

                for line_segment_index in range(len(subpath.points) - 1):
                    associated_faces = [face_indices[face_count_up]]
                    this_face_normal = []
                    if line_segment_index == 0:
                        this_face_normal = self.mesh.face_normals[face_indices[face_count_up]]
                    else:
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

                    mid_pt = (vert1+vert2)/2

                    # plot_normals(visualizer.axs_init, vertices=[np.array(new_ver1)], directions=[np.array(this_face_normal)])

                # check first 2 z values and correct the subpaths' direction
                # plot_path(visualizer.axs_init, np.array(subpath_tool_positions))
                viz_utils.plot_path(viz_utils.visualizer.axs_unord, vertices=np.array(subpath_tool_positions), color='g')

                if (subpath_tool_positions[0][2] > subpath_tool_positions[1][2]) ^ self.direction_flag:
                    subpath_tool_positions.reverse()
                    subpath_tool_normals.reverse()

                all_verts_this_section.append(subpath_tool_positions)
                tool_normals_this_section.append(subpath_tool_normals)

                subpath_tool_positions = np.array(subpath_tool_positions)
                viz_utils.visualizer.axs_unord.scatter(subpath_tool_positions[:, 0],subpath_tool_positions[:, 1],subpath_tool_positions[:, 2], c='g', s=20)
                #viz_utils.plot_normals(viz_utils.visualizer.axs_temp, subpath_tool_positions, subpath_tool_normals, lw=1, hw=0.2)
                viz_utils.plot_path(viz_utils.visualizer.axs_temp, vertices=subpath_tool_positions, color='g', lw=1, hw=0.01)

            """viz_c = 0
            for subpath_tool_positions in all_verts_this_section:
                for viz_i in range(len(subpath_tool_positions)-1):
                    mid_pt = (subpath_tool_positions[viz_i]+subpath_tool_positions[viz_i+1])/2
                    viz_utils.visualizer.axs_temp.text(mid_pt[0], mid_pt[1] + 0.1, mid_pt[2], str(viz_c),
                                                        zdir=None).set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='None'))
                    viz_c += 1"""
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

    def combine_subpaths(self, tool_positions: [[]], tool_normals: [[]], vert_dist_threshold: float,
                         adjacent_tool_normal_angle_threshold: float):
        ele_popped = 0

        print('poses for comb', np.array(tool_positions))
        for i in range(len(tool_positions) - 1):
            inter_vert_dist = LA.norm(np.array(tool_positions[i - ele_popped][-1]) - np.array(
                tool_positions[i + 1 - ele_popped][0]))
            inter_vert_angle = np.arccos(
                np.clip(
                    np.dot(np.array(tool_normals[i - ele_popped][-1]), np.array(tool_normals[i + 1 - ele_popped][0])),
                    -1.0, 1.0))

            if inter_vert_dist < vert_dist_threshold and inter_vert_angle < adjacent_tool_normal_angle_threshold:
                vert_1 = np.array(tool_positions[i + 1 - ele_popped].pop(0))  # remove first vertex in next group
                vert_2 = np.array(tool_positions[i - ele_popped][-1])
                vert_avg = (vert_1+vert_2)*0.5
                tool_positions[i - ele_popped][-1] = vert_avg.tolist() # last vertex of first group should be replaced by the avg
                tool_positions[i - ele_popped] += tool_positions.pop(i + 1 - ele_popped)  # append next group to current group

                norm_1 = tool_normals[i + 1 - ele_popped].pop(0)
                norm_2 = tool_normals[i - ele_popped][-1]

                norm_avg = (norm_1+norm_2)*0.5
                norm_avg /= LA.norm(norm_avg)
                tool_normals[i - ele_popped][-1] = norm_avg.tolist()

                tool_normals[i - ele_popped] += tool_normals.pop(i + 1 - ele_popped)
                # print('popping in after', section_iter, len(tool_positions))
                ele_popped += 1
        print('tool_positions combine_subpaths', len(tool_positions))

    def filter_sample_points(self, samples: [[]], normals: [[]], adjacent_tool_pose_angle_threshold: float,
                             adjacent_vertex_angle_threshold: float, inter_ver_dist_thresh: float):

        popped_indices = []
        new_samples = []
        """for _ in [0, 1]:
            print('len samp', len(samples), np.array(samples))
            ele_popped = 0
            for i, point in enumerate(samples[1:-1]):

                point = np.array(samples[i + 1 - ele_popped])
                # print('samples', samples, i)
                prev_point = np.array(samples[i - ele_popped])
                next_point = np.array(samples[i + 2 - ele_popped])
                a = point - prev_point
                b = next_point - point
                inter_normal_angle = np.arccos(
                    np.clip(np.dot(normals[i - 1 - ele_popped], normals[i - ele_popped]), -1.0, 1.0))
                inter_vert_dist = LA.norm(a)
                inter_vert_angle = np.arccos(np.clip(np.dot(a, b) / (LA.norm(a) * LA.norm(b)), -1.0, 1.0))
                print('int ve', inter_vert_angle, LA.norm(a), LA.norm(b))
                if inter_vert_angle >= adjacent_vertex_angle_threshold or LA.norm(b)==0.0 or LA.norm(a)==0.0: # (inter_normal_angle <= adjacent_tool_pose_angle_threshold) or :  # inter_vert_dist < inter_ver_dist_thresh  or
                    # We dont't threshold inter-vert distances because elimination only depends on normal angles and how collinear the intermediate point is
                    print('popping')
                    samples.pop(i + 1 - ele_popped)
                    normals.pop((i + 1 - ele_popped))
                    popped_indices.append(i + 1)
                    ele_popped += 1
                else:
                    new_samples.append(point)
            print('len samp after', len(samples), np.array(samples))"""

        if len(samples)>2:
            # First remove repeating points
            ele_popped = 0
            for i, point in enumerate(samples[1:]):
                prev_point = np.array(samples[i])
                if LA.norm(np.array(point)-np.array(prev_point) ):
                    samples.pop(i + 1 - ele_popped)
                    normals.pop((i + 1 - ele_popped))
                    ele_popped += 1
            ele_popped = 0
            print('len samp', len(samples), np.array(samples))
            for i, point in enumerate(samples[1:-1]):
                point = np.array(samples[i + 1-ele_popped])
                prev_point = np.array(samples[i-ele_popped])
                next_point = np.array(samples[i + 2-ele_popped])
                a = prev_point- point
                a /= LA.norm(a)
                b = next_point - point
                b /= LA.norm(b)
                inter_vert_angle = np.arccos(np.clip(np.dot(a, b) / (LA.norm(a) * LA.norm(b)), -1.0, 1.0))
                print('int ve', inter_vert_angle, LA.norm(a), LA.norm(b))
                if inter_vert_angle >= adjacent_vertex_angle_threshold or LA.norm(b) == 0.0 or LA.norm(
                        a) == 0.0:  # (inter_normal_angle <= adjacent_tool_pose_angle_threshold) or :  # inter_vert_dist < inter_ver_dist_thresh  or
                    # We dont't threshold inter-vert distances because elimination only depends on normal angles and how collinear the intermediate point is
                    print('popping')
                    samples.pop(i + 1 - ele_popped)
                    normals.pop((i + 1 - ele_popped))
                    popped_indices.append(i + 1)
                    ele_popped += 1
                else:
                    new_samples.append(point)
            print('len samp after', len(samples), np.array(samples))
        return popped_indices