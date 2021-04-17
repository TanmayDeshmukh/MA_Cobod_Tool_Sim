from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import trimesh
import itertools
import copy
from scipy import spatial
from overlap_optimisation import *
from processing_utils import *
import json
from matplotlib.animation import FuncAnimation
import open3d as o3d
import surface_selection_tool
import viz_utils
import seaborn as sns
import warnings

# warnings.filterwarnings('error')

mesh = trimesh.load_mesh('models/wall_type_1_large_angled.STL')

use_eigen_vector_index          = 0
constant_vel                    = 0.25  # m/s
deposition_sim_time_resolution  = 0.05  # s
tool_motion_time_resolution     = 0.5  # s
standoff_dist                   = 0.5  # m
vert_dist_threshold             = 0.05 # m
adjacent_tool_pose_angle_threshold = np.radians(1.0)
adjacent_vertex_angle_threshold = np.radians(1.0)
direction_flag                  = False
number_of_samples               = 1000
surface_sample_viz_size         = 20
tool_pitch_speed_compensation   = True

gun_model = SprayGunModel()

visualizer = viz_utils.Visualizer()

slicing_distance = get_optimal_overlap_distance(gun_model, 0, 0) + gun_model.a/2

get_overlap_profile(gun_model, slicing_distance - gun_model.a/2, 0, 0)
get_1d_overlap_profile(gun_model, slicing_distance - gun_model.a/2, 0, 0, True)

visualizer.mesh_view_adjust(mesh)

faces_mask = surface_selection_tool.get_mask_triangle_indices(mesh)
mesh.update_faces(faces_mask)

# Remove all vertices in the current mesh which are not referenced by a face.
mesh.remove_unreferenced_vertices()
mesh.remove_infinite_values()
mesh.export('surface_only.stl')

# ############### Full model #################
visualizer.draw_mesh(mesh)

# ############################### PCA #################################

covariance_matrix = np.cov(mesh.vertices.T)
eigen_values, eigen_vectors = LA.eig(covariance_matrix)  # returns normalized eig vectors
idx = eigen_values.argsort()[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]

print("Eigenvector after sort: \n", eigen_vectors, "\n")
print("Eigenvalues after sort: \n", eigen_values, "\n")

start = np.min(mesh.vertices, axis=0) + eigen_vectors[:, 2] * (
            eigen_values[2] * 10) + eigen_vectors[:, use_eigen_vector_index] * (slicing_distance- gun_model.a)/2
stop = np.max(mesh.vertices, axis=0) + eigen_vectors[:, use_eigen_vector_index] * slicing_distance
if use_eigen_vector_index == 0:
    start[2] = stop[2] = 0
elif use_eigen_vector_index == 1:
    start[0] = stop[0] = 0
    start[1] = stop[1] = 0
elif use_eigen_vector_index == 2:
    start[1] = stop[1] = 0
length = LA.norm(stop - start)
print('start ', start, stop, length, np.arange(0, length, step=slicing_distance))
print("Eigenvector: \n", eigen_vectors, "\n")

viz_utils.plot_normals(visualizer.axs_init, [start], [eigen_vectors[:, use_eigen_vector_index]], norm_length=length, color='b')
print('eigen_vectors[:,0]', eigen_vectors[:, 0])

# ################################# Slicing #####################################

sections = mesh.section_multiplane(plane_origin=start,
                                   plane_normal=eigen_vectors[:, use_eigen_vector_index],
                                   heights=np.arange(0, length, step=slicing_distance))
print('sections', len(sections))
sections = [s for s in sections if s]
print('sections', len(sections))

d3sections = [section.to_3D() for section in sections]

face_indices = [path.metadata['face_index'] for path in sections]
face_normals = [mesh.face_normals[segment_face_indices] for segment_face_indices in face_indices]
print('mesh attrib', len(mesh.vertex_normals))

vert_iter = 0
section_end_vert_pairs = []
all_tool_locations = []
all_tool_normals = []

# ############################ Ordering, Filtering, Connecting ##############################

for section_iter, section_path_group in enumerate(d3sections):
    face_indices = section_path_group.metadata['face_index']
    new_entities = []
    all_verts_this_section = []
    tool_normals_this_section = []
    direction_flag = not direction_flag
    face_count_up = 0
    for subpath_iter, subpath in enumerate(section_path_group.entities):
        subpath_tool_positions = []
        subpath_tool_normals = []
        for line_segment_index in range(len(subpath.points) - 1):
            this_face_normal = mesh.face_normals[face_indices[face_count_up]]
            face_count_up += 1

            vert1_index, vert2_index = subpath.points[line_segment_index], subpath.points[line_segment_index + 1]
            vert1, vert2 = section_path_group.vertices[vert1_index], section_path_group.vertices[vert2_index]

            new_ver1 = this_face_normal * standoff_dist + vert1
            new_ver2 = this_face_normal * standoff_dist + vert2

            subpath_tool_positions.append([x for x in new_ver1])
            subpath_tool_positions.append([x for x in new_ver2])
            subpath_tool_normals.append(this_face_normal)
            subpath_tool_normals.append(this_face_normal)

            # plot_normals(visualizer.axs_init, vertices=[np.array(new_ver1)], directions=[np.array(this_face_normal)])

        # check first 2 z values and correct the subpaths' direction
        # plot_path(visualizer.axs_init, np.array(subpath_tool_positions))

        if (subpath_tool_positions[0][2] > subpath_tool_positions[1][2]) ^ direction_flag:
            subpath_tool_positions.reverse()
            subpath_tool_normals.reverse()

        all_verts_this_section.append(subpath_tool_positions)
        tool_normals_this_section.append(subpath_tool_normals)

        subpath_tool_positions = np.array(subpath_tool_positions)

    # Correct the order of subpaths first (bubble sorting)
    for i in range(len(all_verts_this_section)):
        for j in range(len(all_verts_this_section) - 1):
            if (all_verts_this_section[j][0][2] > all_verts_this_section[j + 1][0][2]) ^ direction_flag:
                all_verts_this_section[j], all_verts_this_section[j + 1] = all_verts_this_section[j + 1], \
                                                                           all_verts_this_section[j]
                tool_normals_this_section[j], tool_normals_this_section[j + 1] = tool_normals_this_section[j + 1], \
                                                                                 tool_normals_this_section[j]

    # Combine sub-paths if endpoints are close enough. This must be done before removing unnecessary intermediate points
    # because the end points of the sub paths themselves might be unnecessary
    combine_subpaths(all_verts_this_section, tool_normals_this_section, vert_dist_threshold, adjacent_tool_pose_angle_threshold)

    # Remove unnecessary intermediate points
    for vert_group, norms in zip(all_verts_this_section, tool_normals_this_section):
        filter_sample_points(vert_group, norms, adjacent_tool_pose_angle_threshold=adjacent_tool_pose_angle_threshold,
                             adjacent_vertex_angle_threshold=adjacent_vertex_angle_threshold,
                             inter_ver_dist_thresh=vert_dist_threshold)
        vert_group = np.array(vert_group)
        visualizer.axs_init.scatter(vert_group[:, 0], vert_group[:, 1],
                                    vert_group[:, 2], s=2.5, c='r')

    # Extend tool travel outward for better coverage
    for subpath_tool_positions in all_verts_this_section:
        start_direction = np.array(subpath_tool_positions[0])-np.array(subpath_tool_positions[1])
        start_direction /= LA.norm(start_direction)
        start_direction *= gun_model.b
        end_direction = np.array(subpath_tool_positions[-1]) - np.array(subpath_tool_positions[-2])
        end_direction /= LA.norm(end_direction)
        end_direction *= gun_model.b

        # subpath_tool_positions[0] = [sum(x) for x in zip(subpath_tool_positions[0], start_direction.tolist())]
        # subpath_tool_positions[-1] = [sum(x) for x in zip(subpath_tool_positions[0-1], end_direction.tolist())]

    all_tool_normals += tool_normals_this_section
    tool_normals_this_section = -np.array(list(itertools.chain.from_iterable(tool_normals_this_section)))

    viz_utils.plot_normals(visualizer.axs_init, vertices=list(itertools.chain.from_iterable(all_verts_this_section)),
                 directions=tool_normals_this_section)

    # Visualization of activated(g) and deactivated(k) tool travel within this section cut
    for i, ver_group in enumerate(all_verts_this_section):
        plot_path(visualizer.axs_init, vertices=ver_group)
        # plot_path(axs[1][1], vertices=ver_group)

        if i > 0:
            plot_path(visualizer.axs_init, vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]], color='k')
            plot_path(visualizer.final_path_ax, vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]], color='k')
            vert_iter += 1

    all_tool_locations += all_verts_this_section
    all_verts_this_section = list(itertools.chain.from_iterable(all_verts_this_section))
    all_verts_this_section = np.array(all_verts_this_section)
    section_end_vert_pairs += [all_verts_this_section[-1]] if section_iter == 0 else [all_verts_this_section[0],
                                                                                      all_verts_this_section[-1]]

# all_tool_locations = [ [v0 v1 v2 v3 .. vn ] , [v0 v1 v2 v3 .. vm] .. ]
# tool must be ON and move through [v0 v1 v2 v3 .. vn] continuously,
# and off between ..vn of one group and v0.. of the next
# Vert groups may or may not be from the same section

for i in range(int(len(section_end_vert_pairs) / 2)):
    plot_path(visualizer.axs_init, vertices=[section_end_vert_pairs[i * 2], section_end_vert_pairs[i * 2 + 1]], color='k')

plt.draw()
plt.pause(0.001)

print('all_tool_locations\n', all_tool_locations, '\nall_tool_normals\n', all_tool_normals)

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


# ############### Writing to a JSON file ##################
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
    visualizer.final_path_ax.scatter(sorted_intersection_locations[:, 0], sorted_intersection_locations[:, 1],
                               sorted_intersection_locations[:, 2], s=10, color='b', zorder=1500)

    plot_path(visualizer.final_path_ax, continuous_tool_positions)
    viz_utils.plot_normals(visualizer.final_path_ax, continuous_tool_positions, continuous_tool_normals)

    for i, (current_tool_position, current_tool_normal, intersection_location) in enumerate(
            zip(continuous_tool_positions, continuous_tool_normals, sorted_intersection_locations)):

        tool_pos_to_point = current_tool_position-intersection_location
        actual_norm_dist = LA.norm(tool_pos_to_point)
        tool_pos_to_point /= actual_norm_dist
        surface_normal = mesh.face_normals[intersection_index_tri[i]]

        time_scale = 1.0
        if tool_pitch_speed_compensation:
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


def update(frame_number, scatter, deposition_thickness):
    global paint_pass, j,sorted_intersection_locations, continuous_tool_positions, continuous_tool_normals, \
        intersection_index_tri, tool_major_axis_vecs, tool_minor_axis_vecs, sample_tree, mesh, \
        deposition_sim_time_resolution, gun_model

    if paint_pass >= len(total_tool_positions):
        print('\ndeposition_thickness\nmin:', deposition_thickness.min() * 1000, 'mm\nmax',
              deposition_thickness.max() * 1000, 'mm\ndiff: ', (deposition_thickness.max()-deposition_thickness.min())*1000,
              'mm\nstd:', deposition_thickness.std(0) * 1000, '\nmean:', deposition_thickness.mean(0) * 1000,
              '\nmedian:', np.median(deposition_thickness)*1000)
        #sns.distplot(deposition_thickness*1000, ax=visualizer.ax_distrib_hist)
        # plt.draw()
        #plt.show()
        animation.event_source.stop()
        animation.save_count = frame_number
    else:
        if j==0:
            continuous_tool_positions, continuous_tool_normals = all_tool_positions[paint_pass], tool_normals[paint_pass]
            intersection_locations, index_ray, intersection_index_tri = ray_mesh_intersector.intersects_location(
                np.array(continuous_tool_positions),
                np.array(continuous_tool_normals))
            print('\nSorting', intersection_locations)
            sorted_intersection_locations = [loc for loc, _ in
                                             sorted(zip(intersection_locations, index_ray), key=lambda pair: pair[1])]
            print('\nSorted', sorted_intersection_locations)
            continuous_tool_positions, continuous_tool_normals = limit_tool_positions(continuous_tool_positions, np.array(
                sorted_intersection_locations), continuous_tool_normals)
            # plot_path(final_rendering_ax, continuous_tool_positions)
            # viz_utils.plot_normals(final_rendering_ax, continuous_tool_positions, continuous_tool_normals)
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

            #viz_utils.plot_normals(final_rendering_ax, continuous_tool_positions, tool_minor_axis_vecs, norm_length=0.3, color='g')

        # print('new_position', j, 'pass', paint_pass)
        tool_pos_to_point = continuous_tool_positions[j] - sorted_intersection_locations[j]
        actual_norm_dist = LA.norm(tool_pos_to_point)
        # print('intersection_index_tri', intersection_index_tri)
        # print('intersection_index_tri[j]', intersection_index_tri[j])
        surface_normal = mesh.face_normals[intersection_index_tri[j]]
        time_scale = 1.0
        if tool_pitch_speed_compensation:
            time_scale = 1.0/ surface_scaling(gun_model.h, actual_norm_dist, surface_normal, tool_pos_to_point, continuous_tool_normals[j])
        if j%5==0:
            print(f'time_scale {time_scale: .3f}')
            visualizer.ax_distrib_hist.cla()
            # visualizer.ax_distrib_hist.hist(deposition_thickness, color='blue', edgecolor='black', bins='auto', density=False)
            visualizer.ax_distrib_hist.set_xlabel('deposition thickness (mm)')
            binwidth = 0.02
            min_val,max_val = np.min(deposition_thickness)*1000, np.max(deposition_thickness)*1000
            val_width = (max_val - min_val)
            n_bins = int(val_width / binwidth)
            if n_bins==0:
                n_bins=1
            print('bins', n_bins, val_width)
            sns.histplot(deposition_thickness * 1000, kde=True, bins=n_bins, ax=visualizer.ax_distrib_hist) # , binrange=(min_val, max_val)
            arange =  np.arange(min_val , max_val , binwidth)
            print('np.arange(min_val , max_val , binwidth)', arange, 'max', max_val)
            # visualizer.ax_distrib_hist.set_xticks(np.arange(min_val -binwidth/2, max_val +binwidth/2, binwidth))
            #if arange.shape[0] > 0:
            #     visualizer.ax_distrib_hist.set_xlim(0, arange[-1]+binwidth/2)
            plt.draw()
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

    return scatter,
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

animation = FuncAnimation(final_rendering_fig, update, interval= deposition_sim_time_resolution*1000, blit=False,
                          save_count=350, fargs=(scatter, deposition_thickness)) # , cache_frame_data=False, repeat = False)

print('matplotlib.animation.writers', matplotlib.animation.writers.list())
Writer = matplotlib.animation.writers['html']
writer = Writer(fps=15, metadata={'artist':'COBOD'}, bitrate=1800)
# animation.save('paint_simulation.html', writer=writer, progress_callback = \
#     lambda i, n: print(f'Saving frame {i} of {n}'))
# animation.save('image.mp4', fps=20, writer="avconv", codec="libx264")

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
