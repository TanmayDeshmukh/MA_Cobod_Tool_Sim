from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from numpy import linalg as LA
import inspect
import trimesh
import itertools
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

import copy
from viz_utils import *
from spray_gun_model import *
from scipy import spatial
from mpl_toolkits import mplot3d
from overlap_optimisation import *
import json

from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as v

mesh = trimesh.load_mesh('models/wall_type_1_angled.STL')

number_of_samples               = 5000
constant_vel                    = 0.2  # m/s
deposition_sim_time_resolution  = 0.2  # s
tool_motion_time_resolution     = 0.2
standoff_dist                   = 0.2
vert_dist_threshold             = 0.05
adjacent_tool_pose_angle_threshold = np.radians(10.0)
adjacent_vertex_angle_threshold = np.radians(10.0)
direction_flag = False
gun_model = SprayGunModel()
slicing_distance = get_optimal_overlap_distance(gun_model, 0, 0)


fig, axs = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': '3d'})
for axr in axs:
    for ax in axr:
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()
        min_lim = min(mesh.bounds[0, :])
        max_lim = max(mesh.bounds[1, :])
        ax.set_xlim3d(mesh.bounds[0][0] - 0.5, mesh.bounds[1][0] + 0.5)
        ax.set_ylim3d(mesh.bounds[0][1] - 0.5, mesh.bounds[1][1])
        ax.set_zlim3d(mesh.bounds[0][2], mesh.bounds[1][2] + 0.5)

        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis=1))

fig.tight_layout()
fig.subplots_adjust(left=-0.1, right=1.1, top=1.1, bottom=-0.05)

original_mesh = copy.copy(mesh)
scene = trimesh.Scene()
# scene.add_geometry(original_mesh)


################ Full model #################
mplot = mplot3d.art3d.Poly3DCollection(original_mesh.triangles)
mplot.set_alpha(0.5)
mplot.set_facecolor('grey')
mplot.set_edgecolor('black')
mplot.set_sort_zpos(-2)
axs[0][0].add_collection3d(mplot)

faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[1] < -0.1 and normal[0] > -0.5])
print('faces_mask', faces_mask)
mesh.update_faces(faces_mask)
mesh.remove_unreferenced_vertices()
mesh.remove_infinite_values()
# Remove all vertices in the current mesh which are not referenced by a face.
mesh.visual.face_colors = [50, 150, 50, 255]
scene.add_geometry(mesh)

mplot = mplot3d.art3d.Poly3DCollection(mesh.triangles)
# mplot.set_alpha(0.6)
mplot.set_facecolor('cornflowerblue')
# mplot.set_edgecolor('k')
mplot.set_sort_zpos(-1)
axs[0][1].add_collection3d(mplot)

################ PCA ##################

covariance_matrix = np.cov(mesh.vertices.T)
eigen_values, eigen_vectors = LA.eig(covariance_matrix)  # returns normalized eig vectors
idx = eigen_values.argsort()[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]

print("Eigenvector after sort: \n", eigen_vectors, "\n")
print("Eigenvalues after sort: \n", eigen_values, "\n")

start = np.min(mesh.vertices, axis=0)  + eigen_vectors[:, 2] * (
            eigen_values[2] * 10)  + eigen_vectors[:, 0] * slicing_distance / 5
start[2] = 0
stop = np.max(mesh.vertices, axis=0) + eigen_vectors[:, 0] * slicing_distance
stop[2] = 0
length = LA.norm(stop - start)
print('start ', start, stop, length, np.arange(0, length, step=slicing_distance))
print("Eigenvector: \n", eigen_vectors, "\n")
plot_normals(axs[0][0], [start], [eigen_vectors[:, 0]], norm_length=length)
# plot_normals(axs[0][0], [stop ], [-eigen_vectors[:,0]], norm_length = 2)
plot_normals(axs[0][0], [start], [eigen_vectors[:, 1]], norm_length=1, color='g')
plot_normals(axs[0][0], [start], [eigen_vectors[:, 2]], norm_length=1, color='b')

# plt.show()
print('eigen_vectors[:,0]', eigen_vectors[:, 0])
# find a bunch of parallel cross sections
sections = mesh.section_multiplane(plane_origin=start,
                                   plane_normal=eigen_vectors[:, 0],
                                   heights=np.arange(0, length, step=slicing_distance))
sections = [s for s in sections if s]
print('sections', len(sections))

sectionsverts = [s.vertices for s in sections]
d3sections = [section.to_3D() for section in sections]

face_indices = [path.metadata['face_index'] for path in sections]
face_normals = [mesh.face_normals[segment_face_indices] for segment_face_indices in face_indices]


def filter_sample_points(samples: [[]], normals: [[]], adjacent_tool_pose_angle_threshold: float,
                         adjacent_vertex_angle_threshold: float, inter_ver_dist_thresh: float):
    if section_iter == 4:
        print('samples', len(samples), 'normals', len(normals))
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
        if section_iter == 4:
            print('iter vert dist', inter_vert_dist, 'angle', np.degrees(inter_vert_angle))
        if (
                inter_normal_angle < adjacent_tool_pose_angle_threshold) or inter_vert_angle <= adjacent_vertex_angle_threshold:  # inter_vert_dist < inter_ver_dist_thresh  or
            # We dont't threshold inter-vert distances because elimination only depends on normal angles and how collinear the intermediate point is
            if section_iter == 4:
                print('i + 1 - ele_popped', i + 1 - ele_popped, 'ele_popped', popped_indices)
            samples.pop(i + 1 - ele_popped)
            normals.pop((i + 1 - ele_popped))
            popped_indices.append(i + 1)
            ele_popped += 1
    if section_iter == 4:
        print('popped_indices', popped_indices)
    return popped_indices


def angle_between_vectors(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def combine_subpaths(all_verts_this_section, all_normals, vert_dist_threshold, adjacent_tool_pose_angle_threshold):
    ele_popped = 0
    for i in range(len(all_verts_this_section) - 1):
        print('all_verts_this_section', len(all_verts_this_section[i - ele_popped]), len(all_normals[i - ele_popped]))
        inter_vert_dist = LA.norm(np.array(all_verts_this_section[i - ele_popped][-1]) - np.array(
            all_verts_this_section[i + 1 - ele_popped][0]))
        # inter_vert_angle = trimesh.geometry.vector_angle(np.array([all_normals[i][-1], all_normals[i+1][0]]))
        inter_vert_angle = np.arccos(
            np.clip(np.dot(np.array(all_normals[i - ele_popped][-1]), np.array(all_normals[i + 1 - ele_popped][0])),
                    -1.0, 1.0))
        if (inter_vert_dist < vert_dist_threshold and inter_vert_angle < adjacent_tool_pose_angle_threshold):
            if section_iter == 4:
                print('popping in ', section_iter, len(all_verts_this_section))
            all_verts_this_section[i + 1 - ele_popped].pop(0)  # remove first vertex in next group
            all_verts_this_section[i - ele_popped] += all_verts_this_section.pop(
                i + 1 - ele_popped)  # append next group to current group
            all_normals[i + 1 - ele_popped].pop(0)
            all_normals[i - ele_popped] += all_normals.pop(i + 1 - ele_popped)
            # print('popping in after', section_iter, len(all_verts_this_section))
            ele_popped += 1


# filter_sample_points([[0,1,1], [0, 1,0], [0, 2, 0], [0, 3, 0], [0, 2, 1]], np.radians(10.0), 1.2)


print('mesh attrib', len(mesh.vertex_normals))

vert_iter = 0
section_end_vert_pairs = []
all_tool_locations = []
all_tool_normals = []
for section_iter, section_path_group in enumerate(d3sections):
    # section_path_group.show()
    scene.add_geometry(section_path_group)
    # print('\npath_group',type(section_path_group) ,section_path_group, 'path_group attribs:\n',
    # section_path_group.__dict__ , '\n')
    face_indices = section_path_group.metadata['face_index']
    new_entities = []
    all_verts_this_section = []
    all_normals = []
    direction_flag = not direction_flag
    # print('len section_path_group.entities', len(section_path_group.entities), 'section_path_group.metadata[
    # face_index]', len(face_indices), face_indices, '\nlen section_path_group.vertices',
    # len(section_path_group.vertices))
    face_count_up = 0
    if section_iter == 4:
        print('\n section_path_group.entities', len(section_path_group.entities))
    for subpath_iter, subpath in enumerate(section_path_group.entities):
        translated_verts = []
        ori_verts = []
        normals = []
        if section_iter == 4:
            print('subpath.points', len(subpath.points))
        for line_segment_index in range(len(subpath.points) - 1):
            this_face_normal = mesh.face_normals[face_indices[face_count_up]]
            face_count_up += 1

            vert1_index, vert2_index = subpath.points[line_segment_index], subpath.points[line_segment_index + 1]
            vert1, vert2 = section_path_group.vertices[vert1_index], section_path_group.vertices[vert2_index]
            ori_verts.append(vert1), ori_verts.append(vert2)

            new_ver1 = this_face_normal * standoff_dist + vert1
            new_ver2 = this_face_normal * standoff_dist + vert2

            translated_verts.append([x for x in new_ver1])
            translated_verts.append([x for x in new_ver2])
            normals.append(this_face_normal)
            normals.append(this_face_normal)

        # check first 2 z values and correct the subpaths' direction

        if (translated_verts[0][2] > translated_verts[1][2]) ^ direction_flag:
            translated_verts.reverse()
            normals.reverse()

        # if all_verts:
        # plot traversal without spray
        # plot_path(axs[1][0], vertices=[translated_verts[-1], all_verts_this_section[0]], color='k')
        # insert into appropriate place
        all_verts_this_section.append(translated_verts)
        all_normals.append(normals)
        translated_verts = np.array(translated_verts)
        normals = np.array(normals)

    if section_iter == 4:
        temp_iter = 0
        print('direction_flag', direction_flag)
        for i in range(len(all_verts_this_section)):
            plot_path(axs[0][0], all_verts_this_section[i])
            for vertex in all_verts_this_section[i]:
                axs[0][0].text(vertex[0] + 0.05, vertex[1], vertex[2],
                               str(temp_iter), color='r', zorder=2)
                temp_iter += 1
        print('all_verts_this_section', len(all_verts_this_section))

    # Combine subpaths if endpoints are close enough
    combine_subpaths(all_verts_this_section, all_normals, vert_dist_threshold, adjacent_tool_pose_angle_threshold)

    # Correct the order of subpaths first (sorting)
    for i in range(len(all_verts_this_section)):
        for j in range(len(all_verts_this_section) - 1):
            if ((all_verts_this_section[j][0][2] > all_verts_this_section[j + 1][0][2]) ^ direction_flag):
                ##((all_verts_this_section[j][0][2] < all_verts_this_section[j + 1][0][2]) and direction_flag) or \
                #    (all_verts_this_section[j][0][2] > all_verts_this_section[j + 1][0][2] and not direction_flag):
                if section_iter == 4:
                    print('flipping', j, j + 1, 'because', all_verts_this_section[j][-1],
                          all_verts_this_section[j + 1][0])
                all_verts_this_section[j], all_verts_this_section[j + 1] = all_verts_this_section[j + 1], \
                                                                           all_verts_this_section[j]
                all_normals[j], all_normals[j + 1] = all_normals[j + 1], all_normals[j]
        if section_iter == 4:
            temp_iter = 0;
            for k in range(len(all_verts_this_section)):
                for vertex in all_verts_this_section[k]:
                    axs[0][0].text(vertex[0] + 0.15 + i / 10, vertex[1], vertex[2],
                                   str(temp_iter), color='g', zorder=2)
                    temp_iter += 1
    # if section_iter==4:
    #     for i in range(len(all_verts_this_section)):
    #       print('after sorting', len(all_verts_this_section[i]), 'norms', len(all_normals[i]))

    # Combine subpaths if endpoints are close enough. This must be done before removing unnecessary intermediate points
    # because the end points of the sub paths themselves might be unnecessary
    combine_subpaths(all_verts_this_section, all_normals, vert_dist_threshold, adjacent_tool_pose_angle_threshold)

    for vert_group, norms in zip(all_verts_this_section, all_normals):
        plot_path(axs[0][1], vertices=vert_group)

        # plot_normals(axs[0][1], vertices=vert_group, directions=norms)
    # remove unnecessary intermediate points
    for vert_group, norms in zip(all_verts_this_section, all_normals):
        # if section_iter == 4:
        #    print('filtering', np.array(vert_group))
        filter_sample_points(vert_group, norms, adjacent_tool_pose_angle_threshold=adjacent_tool_pose_angle_threshold,
                             adjacent_vertex_angle_threshold=adjacent_vertex_angle_threshold,
                             inter_ver_dist_thresh=vert_dist_threshold)

    all_tool_normals += all_normals
    all_normals = -np.array(list(itertools.chain.from_iterable(all_normals)))

    plot_normals(axs[1][0], vertices=list(itertools.chain.from_iterable(all_verts_this_section)),
                 directions=all_normals)
    plot_normals(axs[1][1], vertices=list(itertools.chain.from_iterable(all_verts_this_section)),
                 directions=all_normals)

    # Visualization of activated(g) and deactivated(k) tool travel within this section cut
    for i, ver_group in enumerate(all_verts_this_section):
        plot_path(axs[1][0], vertices=ver_group)

        if i > 0:
            plot_path(axs[1][0], vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]], color='k')
            plot_path(axs[0][1], vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]], color='k')
        for vertex in ver_group:
            axs[1][0].text(vertex[0], vertex[1], vertex[2],
                           str(vert_iter), color='g', zorder=2)
            vert_iter += 1

    all_tool_locations += all_verts_this_section
    all_verts_this_section = list(itertools.chain.from_iterable(all_verts_this_section))
    all_verts_this_section = np.array(all_verts_this_section)
    section_end_vert_pairs += [all_verts_this_section[-1]] if section_iter == 0 else [all_verts_this_section[0],
                                                                                      all_verts_this_section[-1]]

# all_tool_locations = [ [v0 v1 v2 v3 .. vn ] , [v0 v1 v2 v3 .. vm] .. ]
# tool must be ON and move through [v0 v1 v2 v3 .. vn] continuously, and off between ..vn and v0.. of next group
# Vert groups may or may not be from the same section

for i in range(int(len(section_end_vert_pairs) / 2)):
    plot_path(axs[0][1], vertices=[section_end_vert_pairs[i * 2], section_end_vert_pairs[i * 2 + 1]], color='k')
    plot_path(axs[1][0], vertices=[section_end_vert_pairs[i * 2], section_end_vert_pairs[i * 2 + 1]], color='k')
# path3ds = trimesh.path.path.Path3D(entities=new_entities, vertices = np., metadata = section_path_group.metadata)

print('all_tool_locations\n', all_tool_locations, '\nall_tool_normals\n', all_tool_normals)
# plot the ground plane
xx, yy = np.meshgrid(np.arange(original_mesh.bounds[0][0], original_mesh.bounds[1][0], 0.2),
                     np.arange(original_mesh.bounds[0][1], original_mesh.bounds[1][1], 0.2))
z = np.full((len(xx), len(xx[0])), 0)
axs[0][1].plot_surface(xx, yy, z, alpha=0.5)

combined3d = np.sum(d3sections)
combined = np.sum(sections)

def interpolate_tool_motion(all_tool_locations, all_tool_normals, sample_dist):
    all_tool_positions, tool_normals = [], []
    for position_pair, normal_pair in zip(all_tool_locations, all_tool_normals):
        point_1, point_2 = np.array(position_pair[0]), np.array(position_pair[1])
        movement_direction = (point_2 - point_1)
        movement_dist = LA.norm(movement_direction)
        movement_direction = movement_direction / movement_dist  # normalizing to get only direction vector
        continuous_tool_positions, continuous_tool_normals = [point_1], []
        n_samples = int(movement_dist / sample_dist) + 1
        # print('new vals: ', sample_dist, end =' ')
        while len(continuous_tool_positions) < n_samples:
            next_position = continuous_tool_positions[-1] + movement_direction * sample_dist
            continuous_tool_positions.append(next_position)
            # print(len(continuous_tool_positions))
            # print(next_position, LA.norm(point_2 - continuous_tool_positions[-1]), end=' ')
        #continuous_tool_positions.append(point_2)
        n0, n1 = np.array(normal_pair[0]), np.array(normal_pair[1])

        omega = np.arccos(np.clip(np.dot(n0 / LA.norm(n0), n1 / LA.norm(n1)), -1.0,
                                  1.0))  # Clip so that we dont exceed -1.0, 1.0 due to float arithmatic errors
        # print('\nn0', n0, 'n1', n1, 'omega', omega, np.dot(n0 / LA.norm(n0), n1 / LA.norm(n1)))
        so = np.sin(omega)
        if omega in [0.0, np.inf, np.nan]:
            # Two normals in the same direction, no need for slerp
            continuous_tool_normals = [-normal_pair[0]] * int(n_samples)
            # print('const', len(normals), len(continuous_tool_positions))
        else:
            # Spherical interpolation

            continuous_tool_normals = [-((np.sin((1.0 - t) * omega) / so) * n0 + (np.sin(t * omega) / so) * n1) for t in
                       np.arange(0.0, 1.0, 1.0 / n_samples)]
            # print('slerp', len(normals), len(continuous_tool_positions))
        while len(continuous_tool_normals)>len(continuous_tool_positions):
            continuous_tool_normals.pop(-1)
        if len(continuous_tool_normals)<len(continuous_tool_positions):
            continuous_tool_normals.append(continuous_tool_normals[-1])

        all_tool_positions.append(continuous_tool_positions), tool_normals.append(continuous_tool_normals)
    return all_tool_positions, tool_normals

def affected_points_for_tool_positions(deposition_thickness, sample_tree, sample_face_indexes,
                                       sorted_intersection_locations, tool_positions, tool_normals, tool_major_axes,
                                       gun_model):
    deposition_amount = []
    intensities = []

    # find points within sphere of radius of major axis
    # print('\n affected_points_for_tool_positions', len(sorted_intersection_locations), len(tool_positions), len(tool_normals), len(tool_major_axes))
    # k=1
    for intersection_location, current_tool_position, current_tool_normal, current_tool_major_axis_vec in zip(
            sorted_intersection_locations, tool_positions, tool_normals, tool_major_axes):

        # print(k, end='')
        query_ball_points = sample_tree.query_ball_point(intersection_location, gun_model.a if gun_model.a>gun_model.b else gun_model.b)
        # print('done', len(query_ball_points), query_ball_points)
        # print('.', end='')
        # k += 1
        i = 0
        # print('query_ball_points', len(query_ball_points), query_ball_points)
        for point_index in query_ball_points:
            # print('point_index', point_index, i)
            point = sample_tree.data[point_index]

            tool_pos_to_point = point - current_tool_position
            tool_pos_to_point_dist = LA.norm(tool_pos_to_point)
            tool_pos_to_point /= tool_pos_to_point_dist

            angle_normal_to_point = angle_between_vectors(tool_pos_to_point,
                                                          current_tool_normal)

            # plot_normals(axs[1][1], [current_tool_position], [current_tool_major_axis_vec], norm_length=0.4, color='g')
            # plot_normals(axs[1][1], [current_tool_position], [current_tool_minor_axis_vec], norm_length=0.3, color='b')

            normal_dist_h_dash = np.cos(angle_normal_to_point) * tool_pos_to_point_dist

            rp = tool_pos_to_point*tool_pos_to_point_dist - current_tool_normal * normal_dist_h_dash
            # rp /= LA.norm(rp)
            angle_minor_axis_to_point = angle_between_vectors(rp/LA.norm(rp) , current_tool_minor_axis_vec)
            # rmax = gun_model.a*gun_model.b*np.sqrt((1/ (gun_model.b**2+(gun_model.a*np.tan(angle_major_axis_to_point))**2))**2 +
            #                                       (1/(gun_model.a**2+gun_model.b**2/np.tan(angle_major_axis_to_point)**2))**2)
            # rmax = np.sqrt(((gun_model.a) * np.sin(angle_major_axis_to_point)) ** 2 + (
            #            (gun_model.b) * np.cos(angle_major_axis_to_point)) ** 2)
            # alpha_max = np.arctan(rmax/normal_dist_h_dash)
            # print('dot prod ', np.dot(rp, current_tool_normal))
            d_rp = LA.norm(rp)
            x, y = d_rp * np.sin(angle_minor_axis_to_point), d_rp * np.cos(angle_minor_axis_to_point)
            #print('xy', x, y)
            if gun_model.check_point_validity(x, y):
                #print('valid')
                """
                if normal_dist_h_dash<0.148:
                    plot_normals(axs[1][1], [current_tool_position], [tool_pos_to_point], color='r',
                                 norm_length=tool_pos_to_point_dist)
                    plot_normals(axs[1][1], [current_tool_position], [current_tool_normal],
                                 norm_length=normal_dist_h_dash,
                                 color='g')
                    print('angle_normal_to_point', angle_normal_to_point, 'np.cos()', np.cos(angle_normal_to_point), 'tool_pos_to_point_dist', tool_pos_to_point_dist)
                
                
                print('\ncurrent_tool_minor_axis_vec', current_tool_minor_axis_vec)
                print('current_tool_major_axis_vec', current_tool_major_axis_vec)
                print('angle_normal_to_point', np.degrees(angle_normal_to_point))
                print('normal_dist_h_dash', normal_dist_h_dash)
                
                print('xy', x, y)
                plot_normals(axs[1][1], [current_tool_normal * normal_dist_h_dash + current_tool_position], [rp],
                             norm_length=0.3, color='b')
                
                
                """
                # if angle_normal_to_point < alpha_max:
                # print('alpha_max', np.degrees(alpha_max), 'rmax', rmax, 'normal_dist_h_dash', normal_dist_h_dash,
                #      'angle_normal_to_point', np.degrees(angle_normal_to_point))

                # Estimate deposition thickness for this point
                surface_normal = mesh.face_normals[sample_face_indexes[point_index]]
                # print('surface_normal', surface_normal)

                # print('xy', x, y, 'd_rp', d_rp, 'rp', rp, 'current_tool_normal*normal_dist_h_dash', current_tool_normal*normal_dist_h_dash, 'normal_dist_h_dash', normal_dist_h_dash)
                deposition_at_h = gun_model.deposition_intensity(x, y)
                # if deposition_at_h>0.0:
                #    plot_normals(ax, [current_tool_position], directions=[tool_pos_to_point / LA.norm(tool_pos_to_point)])
                multiplier = ((gun_model.h / tool_pos_to_point_dist) ) * np.dot(surface_normal, tool_pos_to_point)/(np.dot(tool_pos_to_point, -current_tool_normal)**3)
                # print('multiplier', multiplier,'normal_dist_h_dash',normal_dist_h_dash)
                deposition_thickness[point_index] += multiplier * deposition_at_h * deposition_sim_time_resolution
                # print('deposition_thickness[point_index]', deposition_thickness[point_index])
            #    print('invalid')
        # print('done2')
        # print('affected_points', affected_points, '\nintensities', intensities)

    # return affected_points, intensities


# Writing to a JSON file
file_data = []

sample_dist = constant_vel * deposition_sim_time_resolution
all_tool_positions, tool_normals = interpolate_tool_motion(all_tool_locations, all_tool_normals, sample_dist)
for continuous_tool_positions, continuous_tool_normals in zip(all_tool_positions, tool_normals):
    time_stamp = 0
    for i, (current_tool_position, current_tool_normal) in enumerate(
            zip(continuous_tool_positions, continuous_tool_normals)):
        dict = {"time_stamp": time_stamp,
                "z_rotation": 0.0,
                "spray_on": False if i==len(continuous_tool_positions)-1 else True,
                "tool_position": list(current_tool_position),
                "tool_normal": list(current_tool_normal),
                }
        file_data.append(dict)
        time_stamp += tool_motion_time_resolution

with open('tool_positions.json', 'w') as outfile:
    json.dump(file_data, outfile, indent=2)

samples, sample_face_index = trimesh.sample.sample_surface_even(mesh, number_of_samples, radius=None)
deposition_thickness = [0.0] * len(samples)
sample_tree = spatial.KDTree(samples)
ray_mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

############ Create pose list from constant time interval ##############
sim_sample_dist = constant_vel * deposition_sim_time_resolution
all_tool_positions, tool_normals = interpolate_tool_motion(all_tool_locations, all_tool_normals, sim_sample_dist)
j = 0
for continuous_tool_positions, continuous_tool_normals in zip(all_tool_positions, tool_normals):
    intersection_locations, index_ray, intersection_index_tri = ray_mesh_intersector.intersects_location(
        np.array(continuous_tool_positions),
        np.array(continuous_tool_normals))
    print('\n Processing paint pass', j, '/', len(all_tool_positions), ':', len(continuous_tool_positions), end=' ')
    j += 1
    # print('\nsorted_intersection_locations', intersection_locations, index_ray)
    sorted_intersection_locations = [loc for loc, _ in
                                     sorted(zip(intersection_locations, index_ray), key=lambda pair: pair[1])]
    # print('\nsorted_intersection_locations', sorted_intersection_locations)
    tool_major_axis_vecs = []
    for i, (current_tool_position, current_tool_normal) in enumerate(
            zip(continuous_tool_positions, continuous_tool_normals)):  # [:int(len(continuous_tool_positions)/1.5)]
        # set minor axis direction to direction of movement

        current_tool_minor_axis_vec = (continuous_tool_positions[i + 1] - current_tool_position) if i < len(
            continuous_tool_positions) - 1 else current_tool_position - continuous_tool_positions[i - 1]
        current_tool_minor_axis_vec /= LA.norm(current_tool_minor_axis_vec)

        current_tool_major_axis_vec = np.cross(current_tool_minor_axis_vec, current_tool_normal)

        tool_major_axis_vecs.append(current_tool_major_axis_vec)

    affected_points_for_tool_positions(deposition_thickness, sample_tree,
                                       sample_face_index, sorted_intersection_locations,
                                       continuous_tool_positions, continuous_tool_normals,
                                       tool_major_axis_vecs, gun_model)
    sorted_intersection_locations = np.array(sorted_intersection_locations)
    # axs[1][1].scatter(sorted_intersection_locations[:, 0], sorted_intersection_locations[:, 1],
    #                   sorted_intersection_locations[:, 2], s=2.0, c=['r'] * len(sorted_intersection_locations))
    # calculate deposition for each of these points
deposition_thickness = np.array(deposition_thickness)
# np.nan_to_num(deposition_thickness, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
# print('deposition_thickness', deposition_thickness)
print('\ndeposition_thickness min:', deposition_thickness.min() * 1000, ' max', deposition_thickness.max() * 1000,
      ' std:', deposition_thickness.std(0) * 1000, ' mean:', deposition_thickness.mean(0) * 1000)
axs[1][1].scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=0.5, c=deposition_thickness)

plt.show()
