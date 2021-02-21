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
from mpl_toolkits import mplot3d
from matplotlib import pyplot

from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as v

mesh = trimesh.load_mesh('models/wall_type_1_vertical.STL')

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
min_lim = min(mesh.bounds[0, :])
max_lim = max(mesh.bounds[1, :])
ax.set_xlim3d(mesh.bounds[0][0] - 0.5, mesh.bounds[1][0] + 0.5)
ax.set_ylim3d(mesh.bounds[0][1] - 0.5, mesh.bounds[1][1])
ax.set_zlim3d(mesh.bounds[0][2], mesh.bounds[1][2] + 0.5)

limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz']);
ax.set_box_aspect(np.ptp(limits, axis=1))
fig.tight_layout()
fig.subplots_adjust(left=-0.1, right=1.0, top=1.2, bottom=-0.2)

original_mesh = copy.copy(mesh)
scene = trimesh.Scene()
# scene.add_geometry(original_mesh)

""""
################ Full model #################
mplot = mplot3d.art3d.Poly3DCollection(original_mesh.triangles)
mplot.set_alpha(0.5)
mplot.set_facecolor('grey')
#mplot.set_edgecolor('gray')
mplot.set_sort_zpos(-2)
ax.add_collection3d(mplot)
"""

faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[1] < -0.5])
print('faces_mask', faces_mask)
mesh.update_faces(faces_mask)
mesh.visual.face_colors = [50, 150, 50, 255]
scene.add_geometry(mesh)

"""
mplot = mplot3d.art3d.Poly3DCollection(mesh.triangles)
# mplot.set_alpha(0.6)
mplot.set_facecolor('cornflowerblue')
# mplot.set_edgecolor('k')
mplot.set_sort_zpos(-1)
ax.add_collection3d(mplot)
"""

y_extents = mesh.bounds[:, 0]
# slice every .10 model units
y_levels = np.arange(*y_extents, step=0.2)  # -mesh.bounds[0][0]
# https://github.com/mikedh/trimesh/issues/743#issuecomment-642157661
# mesh.show()
# find a bunch of parallel cross sections
print('mesh.bounds', mesh.bounds, 'y_extents', y_extents, 'y_levels', y_levels)
sections = mesh.section_multiplane(plane_origin=[0, 0, 0],
                                   plane_normal=[1, 0, 0],
                                   heights=y_levels)
sections = [s for s in sections if s]
print('sections', sections, sections[0].__dict__)
sectionsverts = [s.vertices for s in sections]
# print('sectionsverts', sectionsverts)
d3sections = [section.to_3D() for section in sections]

face_indices = [path.metadata['face_index'] for path in sections]
d3sectionsverts = [s.vertices for s in d3sections]
# print('face_indices', face_indices)
face_normals = [mesh.face_normals[segment_face_indices] for segment_face_indices in face_indices]


# print('face_normals', face_normals)

def filter_sample_points(samples: [[]], normals: [[]], adjacent_tool_pose_angle_threshold: float,
                         adjacent_vertex_angle_threshold: float, inter_ver_dist_thresh: float):
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
        print('iter vert dist', inter_vert_dist, 'angle', np.degrees(inter_vert_angle))
        if (
                 inter_normal_angle < adjacent_tool_pose_angle_threshold) or inter_vert_angle <= adjacent_vertex_angle_threshold:  # inter_vert_dist < inter_ver_dist_thresh  or
            # We dont't threshold inter-vert distances because elimination only depends on normal angles and how collinear the intermediate point is
            samples.pop(i + 1 - ele_popped)
            normals.pop((i + 1 - ele_popped))
            popped_indices.append(i + 1)
            ele_popped += 1
    print('popped_indices', popped_indices)
    return popped_indices

def angle_between_vectors(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
# filter_sample_points([[0,1,1], [0, 1,0], [0, 2, 0], [0, 3, 0], [0, 2, 1]], np.radians(10.0), 1.2)

print('mesh attrib', len(mesh.vertex_normals))
standoff_dist = 0.2
vert_dist_threshold = 0.05
adjacent_tool_pose_angle_threshold = np.radians(10.0)
adjacent_vertex_angle_threshold = np.radians(10.0)
vert_iter = 0
direction_flag = False
section_end_vert_pairs = []
all_tool_locations = []
all_tool_normals = []
for section_iter, section_path_group in enumerate(d3sections):
    # section_path_group.show()
    scene.add_geometry(section_path_group)
    # print('\npath_group',type(section_path_group) ,section_path_group, 'path_group attribs:\n', section_path_group.__dict__ , '\n')
    face_indices = section_path_group.metadata['face_index']
    new_entities = []
    all_verts = []
    all_normals = []
    all_text_entities = []
    direction_flag = not direction_flag
    # print('len section_path_group.entities', len(section_path_group.entities), 'section_path_group.metadata[face_index]', len(face_indices), face_indices, '\nlen section_path_group.vertices', len(section_path_group.vertices))
    face_count_up = 0
    for subpath_iter, subpath in enumerate(section_path_group.entities):
        translated_verts = []
        ori_verts = []
        normals = []
        ordered_subpath = copy.copy(subpath)
        # print('subpath.points len', len(subpath.points),'entiry', subpath_iter,subpath.__dict__ )
        for line_segment_index in range(len(subpath.points) - 1):
            this_face_normal = mesh.face_normals[face_indices[face_count_up]]
            face_count_up += 1
            normals.append(this_face_normal)
            normals.append(this_face_normal)
            vert1_index = subpath.points[line_segment_index]
            vert1 = section_path_group.vertices[vert1_index]
            vert2_index = subpath.points[line_segment_index + 1]
            vert2 = section_path_group.vertices[vert2_index]
            ori_verts.append(vert1)
            ori_verts.append(vert2)
            new_ver1 = this_face_normal * standoff_dist + vert1
            new_ver2 = this_face_normal * standoff_dist + vert2

            translated_verts.append([x for x in new_ver1])
            translated_verts.append([x for x in new_ver2])

        # check first 2 y values
        if (translated_verts[0][2] > translated_verts[1][2]) ^ direction_flag:
            translated_verts.reverse()
            ordered_subpath.points = np.flip(ordered_subpath.points, axis=0)
            # all_text_entities.append(text_entity)
        # translated_verts = list(set([tuple(x) for x in translated_verts]))

        # print('range(len(translated_verts))', list(range(len(translated_verts))))

        filtered_all_text_entities = []
        # ax.text(filtered_trnaslated_verts[-1][0], filtered_trnaslated_verts[-1][1], filtered_trnaslated_verts[-1][2],
        #        str(vert_iter), color='red')
        '''popped_indices = filter_sample_points(translated_verts, normals, adjacent_tool_pose_angle_threshold,
                                              adjacent_vertex_angle_threshold,
                                              inter_ver_dist_thresh=vert_dist_threshold)
        print('popped_indices', popped_indices)
        print('norms before fil', len(normals), len(translated_verts))
        popped_indices = []
        '''
        filtered_normals = normals
        # filtered_normals = [normal for i, normal in enumerate(normals) if not popped_indices or i not in popped_indices]
        # filtered_trnaslated_verts = [vert for i, vert in enumerate(translated_verts) if not popped_indices or i not in popped_indices]
        print('filtered norm', len(filtered_normals), len(normals), len(translated_verts))

        '''for i in range(1, len(translated_verts)):
            # if sample points are close enough and directions are the same, merge them
            inter_vert_dist = LA.norm(np.array(translated_verts[i])-np.array(translated_verts[i-1]))
            inter_vert_angle = trimesh.geometry.vector_angle(np.array([normals[i-1],normals[i]]))
            inter_vert_angle = np.arccos(np.clip(np.dot(np.array(normals[i-1]), np.array(normals[i])), -1.0, 1.0))
            if (inter_vert_dist>=vert_dist_threshold or inter_vert_angle>=adjacent_tool_pose_angle_threshold) :
                #if section_iter == 4:
                #    print('trimesh.geometry.vector_angle((normals[i-1],normals[i-1]))', inter_vert_angle, inter_vert_dist,  np.array(normals[i]),  np.array(normals[i-1]))
                filtered_trnaslated_verts.append(translated_verts[i])
                filtered_normals.append(normals[i])

                # text_entity = trimesh.path.entities.Text(origin=0, text='point '+str(len(filtered_trnaslated_verts)), height=0.5, vector=1, normal=0, align=None, layer=None)
                # scene.add_geometry(text_entity)
                #text_entity.show(np.array(filtered_trnaslated_verts))
                #text_entity.points = np.array(list(range(len(filtered_trnaslated_verts))))
                #print('text_entity', type(text_entity),text_entity.__dict__, text_entity.points)
                #filtered_all_text_entities.append(text_entity)
        '''
        entity = trimesh.path.entities.Line(
            points=list(range(len(all_verts), len(translated_verts) + len(all_verts))))

        all_text_entities += filtered_all_text_entities

        # if all_verts:
        # plot traversal without spray
        # plot_path(ax, vertices=[all_verts[-1], filtered_trnaslated_verts[0]], color='k')
        # insert into appropriate place
        new_entities.append(entity)
        """for vert_group_index in range(1,len(all_verts)):
            if (filtered_trnaslated_verts[0][1] >= all_verts[vert_group_index-1][1] and filtered_trnaslated_verts[-1][1]<all_verts[vert_group_index][1] and direction_flag) or \
                    (filtered_trnaslated_verts[0][1] <= all_verts[vert_group_index - 1][1] and
                     filtered_trnaslated_verts[-1][1] > all_verts[vert_group_index]) :
                all_verts.insert(vert_group_index, filtered_trnaslated_verts)
                break
                """
        all_verts.append(translated_verts)
        all_normals.append(normals)
        translated_verts = np.array(translated_verts)
        normals = np.array(normals)
        norm_vec = np.column_stack((translated_verts, translated_verts + (normals * .08)))
        # print('trnaslated_verts', np.array(filtered_trnaslated_verts))
        # print('filtered_trnaslated_verts', filtered_trnaslated_verts)
        # print('norm_vec', len(norm_vec), norm_vec, '\nnorm_vec.reshape((-1, 2, 3))', norm_vec.reshape((-1, 2, 3)))
        # print('translated_verts', translated_verts, '\nfiltered_trnaslated_verts\n',filtered_trnaslated_verts,np.transpose(filtered_trnaslated_verts)[0])
        normal_path = trimesh.load_path(norm_vec.reshape((-1, 2, 3)))  # returns Path3D object
        # print('normal_path.entities', len(normal_path.entities), normal_path.entities)
        normal_path.colors = [trimesh.visual.color.to_rgba([255, 0, 0, 255])] * len(normal_path.entities)
        scene.add_geometry(normal_path)
        # plot_normals(ax, vertices= filtered_trnaslated_verts, directions= -filtered_normals)

    # Order subpath first
    for i in range(len(all_verts)):
        for j in range(i):
            if ((not direction_flag) and all_verts[j][0][2] > all_verts[j + 1][-1][2]) or all_verts[j + 1][0][2] > \
                    all_verts[j][-1][2]:
                print('flipping', all_verts[j], all_verts[j + 1])
                all_verts[j], all_verts[j + 1] = all_verts[j + 1], all_verts[j]

    # Combine subpaths if endpoints are close enough. This must be done before removing unnecessary intermediate points
    # because the end points of the sub paths themselves might be unnecessary
    for i in range(len(all_verts) - 1):
        inter_vert_dist = LA.norm(np.array(all_verts[i][-1]) - np.array(all_verts[i + 1][0]))
        # inter_vert_angle = trimesh.geometry.vector_angle(np.array([all_normals[i][-1], all_normals[i+1][0]]))
        inter_vert_angle = np.arccos(np.clip(np.dot(np.array(normals[i - 1]), np.array(normals[i])), -1.0, 1.0))
        if (inter_vert_dist < vert_dist_threshold and inter_vert_angle < adjacent_tool_pose_angle_threshold):
            print('popping in ', section_iter, len(all_verts))
            all_verts[i + 1].pop(0)
            all_verts[i] += all_verts.pop(i + 1)
            all_normals[i + 1].pop(0)
            all_normals[i] += all_normals.pop(i + 1)
            print('popping in after', section_iter, len(all_verts))

    # remove unnecessary intermediate points
    for vert_group, norms in zip(all_verts, all_normals):
        #if section_iter == 4:
        #    print('filtering', np.array(vert_group))
        filter_sample_points(vert_group, norms, adjacent_tool_pose_angle_threshold=adjacent_tool_pose_angle_threshold,
                             adjacent_vertex_angle_threshold=adjacent_vertex_angle_threshold,
                             inter_ver_dist_thresh=vert_dist_threshold)

    all_tool_normals += all_normals
    all_normals = -np.array(list(itertools.chain.from_iterable(all_normals)))
    plot_normals(ax, vertices=list(itertools.chain.from_iterable(all_verts)), directions=all_normals)

    # Visualization of activated(g) and deactivated(k) tool travel
    for i, ver_group in enumerate(all_verts):
        #plot_path(ax, vertices=ver_group)
        if i > 0:
            plot_path(ax, vertices=[all_verts[i - 1][-1], all_verts[i][0]], color='k')
        for vertex in ver_group:
            ax.text(vertex[0], vertex[1], vertex[2],
                    str(vert_iter), color='g', zorder=2)
            vert_iter += 1
    all_tool_locations += all_verts

    all_verts = list(itertools.chain.from_iterable(all_verts))

    all_verts = np.array(all_verts)

    section_end_vert_pairs += [all_verts[-1]] if section_iter == 0 else [all_verts[0], all_verts[-1]]

for i in range(int(len(section_end_vert_pairs) / 2)):
    plot_path(ax, vertices=[section_end_vert_pairs[i * 2], section_end_vert_pairs[i * 2 + 1]], color='k')
# path3ds = trimesh.path.path.Path3D(entities=new_entities, vertices = np., metadata = section_path_group.metadata)

print('all_tool_locations\n', all_tool_locations, '\nall_tool_normals\n', all_tool_normals)
# plot the ground plane
xx, yy = np.meshgrid(np.arange(original_mesh.bounds[0][0], original_mesh.bounds[1][0], 0.2),
                     np.arange(original_mesh.bounds[0][1], original_mesh.bounds[1][1], 0.2))
z = np.full((len(xx), len(xx[0])), 0)
ax.plot_surface(xx, yy, z, alpha=0.5)


combined3d = np.sum(d3sections)
combined = np.sum(sections)

constant_vel = 0.5 # m/s
deposition_sim_time_resolution = 0.1 #s
sample_dist = constant_vel*deposition_sim_time_resolution
# Create pose list from constant time interval
all_tool_positions = []
tool_normals = []
for position_pair, normal_pair in zip(all_tool_locations, all_tool_normals):
    point_1, point_2 = np.array(position_pair[0]), np.array(position_pair[1])
    movement_direction = (point_2-point_1)
    movement_dist = LA.norm(movement_direction)
    movement_direction = movement_direction/movement_dist # normalizing to get only direction vector
    continuous_tool_positions, continuous_tool_normals = [point_1], []
    n_samples = int(movement_dist / sample_dist)+2
    #print('new vals: ', sample_dist, end =' ')
    while len(continuous_tool_positions)<n_samples-1:
        next_position = continuous_tool_positions[-1]+movement_direction*sample_dist
        continuous_tool_positions.append(next_position)
        # print(len(continuous_tool_positions))
        #print(next_position, LA.norm(point_2 - continuous_tool_positions[-1]), end=' ')
    continuous_tool_positions.append(point_2)
    n0, n1 = np.array(normal_pair[0]), np.array(normal_pair[1])

    omega = np.arccos(np.clip(np.dot(n0/LA.norm(n0) , n1/LA.norm(n1)), -1.0, 1.0)) # Clip so that we dont exceed -1.0, 1.0 due to float arithmatic errors
    print('\nn0', n0, 'n1', n1, 'omega', omega, np.dot(n0/LA.norm(n0) , n1/LA.norm(n1)))
    so = np.sin(omega)
    if omega in [0.0, np.inf, np.nan] :
        print('SAME')
        # Two normals in the same direction, no need for slerp
        normals = [-normal_pair[0]]*int(n_samples)
    else:
        print('not the same')
        # Spherical interpolation
        normals = [-((np.sin((1.0 - t)*omega)/so)*n0 + (np.sin(t*omega) / so)*n1) for t in np.arange(0.0, 1.0, 1.0/n_samples)]
    continuous_tool_normals = normals
    print('new_norms', len(normals), len(continuous_tool_positions), n_samples, normals)
    #plot_normals(ax=ax, vertices=continuous_tool_positions, directions=normals)
    # plot_path(ax, continuous_tool_positions)
    #print('cont tool pos', continuous_tool_positions)

    all_tool_positions.append(continuous_tool_positions), tool_normals.append(continuous_tool_normals)


# combined3d.show()
# scene.show()

number_of_samples = 5000
samples, face_index = trimesh.sample.sample_surface_even(mesh, number_of_samples, radius=None)
deposition_thickness = [0.0]*len(samples)
# ax.scatter(samples[:,0], samples[:, 1], samples[:, 2], s=0.1, c=deposition_thickness)
print(face_index)

gun_model = SprayGunModel()

def affected_points_for_tool_position(points, face_indexes, current_tool_position, current_tool_normal, current_tool_major_axis_vec, gun_model):
    affected_points = []
    deposition_amount = []
    intensities = []
    for i, point in enumerate(points):
        tool_pos_to_point = point-current_tool_position
        tool_pos_to_point_dist = LA.norm(tool_pos_to_point)
        # tool_pos_to_point /= LA.norm(tool_pos_to_point)
        angle_normal_to_point = angle_between_vectors(tool_pos_to_point/LA.norm(tool_pos_to_point), current_tool_normal)
        normal_dist_h_dash = np.cos(angle_normal_to_point)*tool_pos_to_point_dist
        rp = tool_pos_to_point-current_tool_normal*normal_dist_h_dash


        angle_major_axis_to_point = angle_between_vectors(rp/LA.norm(rp), current_tool_major_axis_vec)


        rmax = gun_model.a*gun_model.b*np.sqrt((1/ (gun_model.b**2+(gun_model.a*np.tan(angle_major_axis_to_point))**2))**2 +
                                               (1/(gun_model.a**2+gun_model.b**2/np.tan(angle_major_axis_to_point)**2))**2)
        rmax = np.sqrt(((gun_model.a) * np.sin(angle_major_axis_to_point)) ** 2 + (
                    (gun_model.b) * np.cos(angle_major_axis_to_point)) ** 2)
        alpha_max = np.arctan(rmax/normal_dist_h_dash)
        d_rp = LA.norm(rp)
        x, y = d_rp * np.sin(angle_major_axis_to_point), d_rp * np.cos(angle_major_axis_to_point)

        if gun_model.check_point_validity(x, y):
        #if angle_normal_to_point < alpha_max:
            #print('alpha_max', np.degrees(alpha_max), 'rmax', rmax, 'normal_dist_h_dash', normal_dist_h_dash,
            #      'angle_normal_to_point', np.degrees(angle_normal_to_point))

            affected_points.append(i)
            # Estimate deposition thickness for this point
            surface_normal = mesh.face_normals[face_indexes[i]]
            # print('surface_normal', surface_normal)

            # print('xy', x, y, 'd_rp', d_rp, 'rp', rp, 'current_tool_normal*normal_dist_h_dash', current_tool_normal*normal_dist_h_dash, 'normal_dist_h_dash', normal_dist_h_dash)
            deposition_at_h = gun_model.deposition_intensity(x, y)
            #if deposition_at_h>0.0:
            #    plot_normals(ax, [current_tool_position], directions=[tool_pos_to_point / LA.norm(tool_pos_to_point)])
            multiplier = ((gun_model.h/normal_dist_h_dash)**2) * np.dot(surface_normal, -current_tool_normal) * deposition_sim_time_resolution
            intensities.append(deposition_at_h*multiplier)

    #print('affected_points', affected_points, '\nintensities', intensities)

    return affected_points, intensities


print('all_tool_positions', all_tool_positions, '\ntool_normals', tool_normals)

for continuous_tool_positions,continuous_tool_normals  in zip(all_tool_positions[3:5], tool_normals[3:5]):
    for current_tool_position, current_tool_normal in zip(continuous_tool_positions, continuous_tool_normals):
        # find the points affected by this tool position
        current_tool_minor_axis_vec = np.array([0,0,-1])
        current_tool_major_axis_vec = np.cross(current_tool_minor_axis_vec, current_tool_normal)
        affected_points, intensities =affected_points_for_tool_position(samples, face_index, current_tool_position, current_tool_normal, current_tool_major_axis_vec, gun_model)
        for index, intensity in zip(affected_points, intensities):
            deposition_thickness[index] += intensity
            if intensity<0:
                print('in', intensity)
        # calculate deposition for each of these points
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=0.5, c=deposition_thickness)
#ax.plot_surface(samples[:, 0], samples[:, 1], samples[:, 2], c=deposition_thickness)
#ax.plot_trisurf(samples[:, 0], samples[:, 1], samples[:, 2])
plt.show()
def displace_segment_along_vector(path3D, vector, distance):
    return vector * distance + path3D
