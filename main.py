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
import matplotlib

import copy
from viz_utils import *
from spray_gun_model import *
from scipy import spatial
from mpl_toolkits import mplot3d
from overlap_optimisation import *
from processing_utils import *
import json
from matplotlib.animation import FuncAnimation
import open3d as o3d
import mplcursors

mesh = trimesh.load_mesh('models/wall_type_1_angled.STL')

constant_vel                    = 0.3  # m/s
deposition_sim_time_resolution  = 0.05  # s
tool_motion_time_resolution     = 0.5  # s
standoff_dist                   = 0.5  # m
vert_dist_threshold             = 0.05 # m
adjacent_tool_pose_angle_threshold = np.radians(10.0)
adjacent_vertex_angle_threshold = np.radians(10.0)
direction_flag                  = False
number_of_samples               = 2000
surface_sample_viz_size         = 20

gun_model = SprayGunModel()
canvas, X_grid, Y_grid = gun_model.get_deposition_canvas(np.radians(0))
# gun_model.visualize_deposition(canvas, X_grid, Y_grid)
slicing_distance = get_optimal_overlap_distance(gun_model, 0, 0) + gun_model.a/2
get_overlap_profile(gun_model, slicing_distance- gun_model.a/2, 0, 0)
get_1d_overlap_profile(gun_model, slicing_distance- gun_model.a/2, 0, 0, True)

fig_init, axs_init = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
fig_init.canvas.set_window_title('Desired Path')
final_path_fig, final_path_ax = plt.subplots(subplot_kw={'projection': '3d'})
final_path_fig.tight_layout()
final_path_fig.subplots_adjust(left=-0.1, right=1.1, top=1.1, bottom=-0.05)
final_path_fig.canvas.set_window_title('Final Path')

for ax in [axs_init, final_path_ax]:
    #for ax in axr:
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

fig_init.tight_layout()
fig_init.subplots_adjust(left=-0.05, right=1.05, top=1.1, bottom=-0.05)


original_mesh = copy.copy(mesh)
faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[1] < -0.1 and normal[0] > -0.5])
print('faces_mask', faces_mask)
mesh.update_faces(faces_mask)
mesh.remove_unreferenced_vertices()
mesh.remove_infinite_values()
mesh.export('surface_only.stl')
# Remove all vertices in the current mesh which are not referenced by a face.
mesh.visual.face_colors = [50, 150, 50, 255]
# scene.add_geometry(mesh)

# ############### Full model #################
mplot = mplot3d.art3d.Poly3DCollection(mesh.triangles)
mplot.set_alpha(0.5)
mplot.set_facecolor('grey')
# mplot.set_edgecolor('black')
mplot.set_sort_zpos(-2)
axs_init.add_collection3d(mplot)

mplot = mplot3d.art3d.Poly3DCollection(mesh.triangles)
# mplot.set_alpha(0.6)
mplot.set_facecolor('cornflowerblue')
# mplot.set_edgecolor('k')
mplot.set_sort_zpos(-1)
final_path_ax.add_collection3d(mplot)

# ############################### PCA #################################

covariance_matrix = np.cov(mesh.vertices.T)
eigen_values, eigen_vectors = LA.eig(covariance_matrix)  # returns normalized eig vectors
idx = eigen_values.argsort()[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]

print("Eigenvector after sort: \n", eigen_vectors, "\n")
print("Eigenvalues after sort: \n", eigen_values, "\n")

start = np.min(mesh.vertices, axis=0) + eigen_vectors[:, 2] * (
            eigen_values[2] * 10) + eigen_vectors[:, 0] * (slicing_distance- gun_model.a)/2
stop = np.max(mesh.vertices, axis=0) + eigen_vectors[:, 0] * slicing_distance
start[2] = stop[2] = 0
length = LA.norm(stop - start)
print('start ', start, stop, length, np.arange(0, length, step=slicing_distance))
print("Eigenvector: \n", eigen_vectors, "\n")

plot_normals(axs_init, [start], [eigen_vectors[:, 0]], norm_length=length, color='b')
# plot_normals(axs_init, [start], [eigen_vectors[:, 1]], norm_length=1, color='g')
# plot_normals(axs_init, [start], [eigen_vectors[:, 2]], norm_length=1, color='b')
print('eigen_vectors[:,0]', eigen_vectors[:, 0])

# ################################# Slicing #####################################
sections = mesh.section_multiplane(plane_origin=start,
                                   plane_normal=eigen_vectors[:, 0],
                                   heights=np.arange(0, length, step=slicing_distance))
print('sections', len(sections))
sections = [s for s in sections if s]
print('sections', len(sections))

sectionsverts = [s.vertices for s in sections]
d3sections = [section.to_3D() for section in sections]

face_indices = [path.metadata['face_index'] for path in sections]
face_normals = [mesh.face_normals[segment_face_indices] for segment_face_indices in face_indices]
print('mesh attrib', len(mesh.vertex_normals))

vert_iter = 0
section_end_vert_pairs = []
all_tool_locations = []
all_tool_normals = []

# ############################ Ordering Filtering Connecting ##############################

for section_iter, section_path_group in enumerate(d3sections):
    face_indices = section_path_group.metadata['face_index']
    new_entities = []
    all_verts_this_section = []
    all_normals = []
    direction_flag = not direction_flag
    face_count_up = 0
    for subpath_iter, subpath in enumerate(section_path_group.entities):
        translated_verts = []
        ori_verts = []
        normals = []
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
        plot_path(axs_init, np.array(translated_verts))

        if (translated_verts[0][2] > translated_verts[1][2]) ^ direction_flag:
            translated_verts.reverse()
            normals.reverse()

        all_verts_this_section.append(translated_verts)
        all_normals.append(normals)
        translated_verts = np.array(translated_verts)
        normals = np.array(normals)

        axs_init.scatter(translated_verts[:,0], translated_verts[:,1], translated_verts[:,2], s=2.5, c='r')
    if section_iter == 4:
        temp_iter = 0
        print('direction_flag', direction_flag)
        """for i in range(len(all_verts_this_section)):
            plot_path(axs[0][0], all_verts_this_section[i])
            for vertex in all_verts_this_section[i]:
                axs[0][0].text(vertex[0] + 0.05, vertex[1], vertex[2],
                               str(temp_iter), color='r', zorder=2)
                temp_iter += 1"""
        print('all_verts_this_section', len(all_verts_this_section))

    # Combine subpaths if endpoints are close enough
    combine_subpaths(all_verts_this_section, all_normals, vert_dist_threshold, adjacent_tool_pose_angle_threshold)

    # Correct the order of subpaths first (sorting)
    for i in range(len(all_verts_this_section)):
        for j in range(len(all_verts_this_section) - 1):
            if (all_verts_this_section[j][0][2] > all_verts_this_section[j + 1][0][2]) ^ direction_flag:
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
                    axs_init.text(vertex[0] + 0.15 + i / 10, vertex[1], vertex[2],
                                   str(temp_iter), color='g', zorder=2)
                    temp_iter += 1

    # Combine sub-paths if endpoints are close enough. This must be done before removing unnecessary intermediate points
    # because the end points of the sub paths themselves might be unnecessary
    combine_subpaths(all_verts_this_section, all_normals, vert_dist_threshold, adjacent_tool_pose_angle_threshold)

    # for vert_group, norms in zip(all_verts_this_section, all_normals):
    #     plot_path(axs_init[1], vertices=vert_group)

    # plot_normals(axs_init[1], vertices=vert_group, directions=norms)
    # remove unnecessary intermediate points
    for vert_group, norms in zip(all_verts_this_section, all_normals):
        filter_sample_points(vert_group, norms, adjacent_tool_pose_angle_threshold=adjacent_tool_pose_angle_threshold,
                             adjacent_vertex_angle_threshold=adjacent_vertex_angle_threshold,
                             inter_ver_dist_thresh=vert_dist_threshold)

    all_tool_normals += all_normals
    all_normals = -np.array(list(itertools.chain.from_iterable(all_normals)))

    plot_normals(axs_init, vertices=list(itertools.chain.from_iterable(all_verts_this_section)),
                 directions=all_normals)
    # plot_normals(axs[1][1], vertices=list(itertools.chain.from_iterable(all_verts_this_section)),
    #              directions=all_normals)

    # Visualization of activated(g) and deactivated(k) tool travel within this section cut
    for i, ver_group in enumerate(all_verts_this_section):
        plot_path(axs_init, vertices=ver_group)
        # plot_path(axs[1][1], vertices=ver_group)

        if i > 0:
            plot_path(axs_init, vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]], color='k')
            # plot_path(axs[1][1], vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]], color='k')
        for vertex in ver_group:
            axs_init.text(vertex[0], vertex[1], vertex[2],
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
    # plot_path(axs[1][1], vertices=[section_end_vert_pairs[i * 2], section_end_vert_pairs[i * 2 + 1]], color='k')
    plot_path(axs_init, vertices=[section_end_vert_pairs[i * 2], section_end_vert_pairs[i * 2 + 1]], color='k')

plt.draw()
plt.pause(0.001)

print('all_tool_locations\n', all_tool_locations, '\nall_tool_normals\n', all_tool_normals)
# plot the ground plane
xx, yy = np.meshgrid(np.arange(original_mesh.bounds[0][0], original_mesh.bounds[1][0], 0.2),
                     np.arange(original_mesh.bounds[0][1], original_mesh.bounds[1][1], 0.2))
z = np.full((len(xx), len(xx[0])), 0)
final_path_ax.plot_surface(xx, yy, z, alpha=0.5)


final_rendering_fig, final_rendering_ax = plt.subplots(subplot_kw={'projection': '3d'})
final_rendering_fig.tight_layout()
final_rendering_fig.subplots_adjust(left=-0.1, right=1.1, top=1.1, bottom=-0.05)

# Sample points on surface for simulation
samples, sample_face_indexes = trimesh.sample.sample_surface_even(mesh, number_of_samples, radius=None)
deposition_thickness = [0.0] * len(samples)
sample_tree = spatial.KDTree(samples)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(samples)
o3d.io.write_point_cloud("wall_surface.pcd", pcd)

# ################################# Calculate paint passes #################################
deposition_thickness = np.array(deposition_thickness)
scatter = final_rendering_ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=surface_sample_viz_size, picker = 2)#, c=deposition_thickness, cmap='coolwarm')


# Writing to a JSON file
file_data = []
ray_mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
sample_dist = constant_vel * tool_motion_time_resolution
all_tool_positions, tool_normals = interpolate_tool_motion(all_tool_locations, all_tool_normals, sample_dist)
total_time_count = 0
for continuous_tool_positions, continuous_tool_normals in zip(all_tool_positions, tool_normals):
    time_stamp = 0
    intersection_locations, index_ray, intersection_index_tri = ray_mesh_intersector.intersects_location(
        np.array(continuous_tool_positions),
        np.array(continuous_tool_normals))
    sorted_intersection_locations = [loc for loc, _ in
                                     sorted(zip(intersection_locations, index_ray), key=lambda pair: pair[1])]
    continuous_tool_positions, continuous_tool_normals = limit_tool_positions(continuous_tool_positions, np.array(
        sorted_intersection_locations), continuous_tool_normals)
    sorted_intersection_locations = np.array(sorted_intersection_locations)
    final_rendering_ax.scatter(sorted_intersection_locations[:, 0], sorted_intersection_locations[:, 1],
                               sorted_intersection_locations[:, 2], s=17, color='r')

    plot_path(final_path_ax, continuous_tool_positions)
    plot_normals(final_path_ax, continuous_tool_positions, continuous_tool_normals)

    for i, (current_tool_position, current_tool_normal, intersection_location) in enumerate(
            zip(continuous_tool_positions, continuous_tool_normals, sorted_intersection_locations)):

        tool_pos_to_point = current_tool_position-intersection_location
        actual_norm_dist = LA.norm(tool_pos_to_point)
        tool_pos_to_point /= actual_norm_dist
        surface_normal = mesh.face_normals[intersection_index_tri[i]]

        time_scale = 1.0 / surface_scaling(gun_model.h, actual_norm_dist, surface_normal, tool_pos_to_point,
                                           current_tool_normal)
        dict = {"time_stamp": time_stamp*actual_norm_dist/standoff_dist,
                "z_rotation": 0.0,
                "spray_on": False if i==len(continuous_tool_positions)-1 else True,
                "tool_position": list(current_tool_position),
                "tool_normal": list(current_tool_normal),
                }
        file_data.append(dict)
        time_stamp += tool_motion_time_resolution
    total_time_count += time_stamp
with open('tool_positions.json', 'w') as outfile:
    json.dump(file_data, outfile, indent=2)


sim_sample_dist = constant_vel * deposition_sim_time_resolution

# ########### Create pose list from constant time interval ##############
all_tool_positions, tool_normals = interpolate_tool_motion(all_tool_locations, all_tool_normals, sim_sample_dist)

total_tool_positions = [len(continuous_tool_positions) for continuous_tool_positions in all_tool_positions]
paint_pass, j = 0, 0

sorted_intersection_locations = []
continuous_tool_positions = []
continuous_tool_normals = []
tool_major_axis_vecs = []
tool_minor_axis_vecs = []
intersection_index_tri = -1
def update(frame_number):
    global paint_pass, j,sorted_intersection_locations, continuous_tool_positions, continuous_tool_normals, \
        intersection_index_tri, tool_major_axis_vecs, tool_minor_axis_vecs, deposition_thickness, sample_tree, mesh, scatter, deposition_sim_time_resolution, gun_model

    if paint_pass >= len(total_tool_positions):
        print('\ndeposition_thickness\nmin:', deposition_thickness.min() * 1000, 'mm\nmax',
              deposition_thickness.max() * 1000, 'mm\ndiff: ', (deposition_thickness.max()-deposition_thickness.min())*1000,
              'mm\nstd:', deposition_thickness.std(0) * 1000, '\nmean:', deposition_thickness.mean(0) * 1000, '\n')
        animation.event_source.stop()
    else:
        if j==0:
            continuous_tool_positions, continuous_tool_normals = all_tool_positions[paint_pass], tool_normals[paint_pass]
            intersection_locations, index_ray, intersection_index_tri = ray_mesh_intersector.intersects_location(
                np.array(continuous_tool_positions),
                np.array(continuous_tool_normals))
            # print('\nSorting', total_tool_positions[paint_pass])
            sorted_intersection_locations = [loc for loc, _ in
                                             sorted(zip(intersection_locations, index_ray), key=lambda pair: pair[1])]
            continuous_tool_positions, continuous_tool_normals = limit_tool_positions(continuous_tool_positions, np.array(
                sorted_intersection_locations), continuous_tool_normals)
            # plot_path(final_rendering_ax, continuous_tool_positions)
            # plot_normals(final_rendering_ax, continuous_tool_positions, continuous_tool_normals)
            tool_major_axis_vecs, tool_minor_axis_vecs = [], []
            # print('Estim axis vecs')
            current_tool_minor_axis_vec=[]
            current_tool_major_axis_vec=[]
            for pos_index, (current_tool_position, current_tool_normal) in enumerate(
                    zip(continuous_tool_positions, continuous_tool_normals)):
                # set minor axis direction to direction of movement

                if pos_index < len(continuous_tool_positions)-1:
                    # if angle_between_vectors(current_tool_normal, continuous_tool_normals[pos_index+1]) > 0:
                    current_tool_minor_axis_vec = (continuous_tool_positions[pos_index + 1] - current_tool_position)+continuous_tool_normals[pos_index+1] - current_tool_normal
                    current_tool_minor_axis_vec /= LA.norm(current_tool_minor_axis_vec)

                    current_tool_minor_axis_vec /= LA.norm(current_tool_minor_axis_vec)

                    current_tool_major_axis_vec = np.cross(current_tool_minor_axis_vec, current_tool_normal)
                tool_major_axis_vecs.append(current_tool_major_axis_vec)
                tool_minor_axis_vecs.append(current_tool_minor_axis_vec)

            #plot_normals(final_rendering_ax, continuous_tool_positions, tool_minor_axis_vecs, norm_length=0.3, color='g')

        # print('new_position', j, 'pass', paint_pass)
        tool_pos_to_point = continuous_tool_positions[j] - sorted_intersection_locations[j]
        actual_norm_dist = LA.norm(tool_pos_to_point)
        # print('intersection_index_tri', intersection_index_tri)
        # print('intersection_index_tri[j]', intersection_index_tri[j])
        surface_normal = mesh.face_normals[intersection_index_tri[j]]
        time_scale = 1.0/ surface_scaling(gun_model.h, actual_norm_dist, surface_normal, tool_pos_to_point, continuous_tool_normals[j])
        if j%3==0:
            print(f'time_scale {time_scale: .3f}')
        # time_scale = actual_norm_dist/standoff_dist
        # print('time_scale', time_scale, 'actual_norm_dist', actual_norm_dist, 'standoff_dist', standoff_dist)
        animation.event_source.interval = deposition_sim_time_resolution*1000*time_scale
        affected_points_for_tool_position(deposition_thickness, sample_tree, mesh,
                                           sample_face_indexes, sorted_intersection_locations[j],
                                           continuous_tool_positions[j], continuous_tool_normals[j],
                                           tool_major_axis_vecs[j], tool_minor_axis_vecs[j],
                                           gun_model, deposition_sim_time_resolution*time_scale, scatter)

        j += 1
        if j >= total_tool_positions[paint_pass]:
            paint_pass +=1
            j=0


""" 
for continuous_tool_positions, continuous_tool_normals in zip(all_tool_positions[:], tool_normals[:]):
    

    print('\nfiltering')
    continuous_tool_positions, continuous_tool_normals = limit_tool_positions(continuous_tool_positions, np.array(sorted_intersection_locations), continuous_tool_normals)
    print('Filtered')
    affected_points_for_tool_positions(deposition_thickness, sample_tree, mesh,
                                       sample_face_index, sorted_intersection_locations,
                                       continuous_tool_positions, continuous_tool_normals,
                                       tool_major_axis_vecs, tool_minor_axis_vecs,
                                       gun_model, deposition_sim_time_resolution, scatter)

print('deposition_thickness', deposition_thickness)
"""

# scatter.set_clim(vmin=min(deposition_thickness), vmax=max(deposition_thickness))
# scatter.set_color()



final_rendering_fig.canvas.set_window_title('Paint sim')

animation = FuncAnimation(final_rendering_fig, update, interval= deposition_sim_time_resolution*1000, blit=False)
print('\n Cumulative spray time', total_time_count)
# mplcursors.cursor(hover=True)
# cursor = mplcursors.cursor(scatter)
# cursor.connect(
#    "add", lambda sel: sel.annotation.set_text(f'{sel.target.index}: {deposition_thickness[sel.target.index]*1000.0 :.5f} \n {max(deposition_thickness)}'))

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)

# final_rendering_fig.canvas.mpl_connect('pick_event', onpick)


plt.show()
print('after plot')
