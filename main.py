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
from trajectory_generation import TrajectoryGenerator
import json
from matplotlib.animation import FuncAnimation
import open3d as o3d
import surface_selection_tool
import viz_utils
import seaborn as sns
import warnings

# warnings.filterwarnings('error')
stl_file = 'wall_type_1_angled.STL' # wall_type_1_angled.STL wall_type_2_vertical
mesh = trimesh.load_mesh('models/'+stl_file)

use_eigen_vector_index          = 0
constant_vel                    = 0.6  # m/s
deposition_sim_time_resolution  = 0.1  # s
tool_motion_time_resolution     = 0.5 # s
standoff_dist                   = 0.5  # m

number_of_samples               = 5000
surface_sample_viz_size         = 7
tool_pitch_speed_compensation   = False

tool_limits = [0, 2], [0.4, 1.6], [0.7, 1.0] # X, Y, Z
# tool_limits = [-5.0, 5], [-5.0, 5], [-5.0, 5.0]

gun_model = SprayGunModel()

starting_slice_offset           =  gun_model.a/3


slicing_distance = get_optimal_overlap_distance(gun_model, 0, 0) + gun_model.a/2

get_overlap_profile(gun_model, slicing_distance - gun_model.a/2, 0, 0)
get_1d_overlap_profile(gun_model, slicing_distance - gun_model.a/2, 0, 0, True)

viz_utils.open_figures()
viz_utils.visualizer.mesh_view_adjust(mesh)

faces_mask = surface_selection_tool.get_mask_triangle_indices(mesh)
mesh.update_faces(faces_mask)
# Remove all vertices in the current mesh which are not referenced by a face.
mesh.remove_unreferenced_vertices()
mesh.remove_infinite_values()
mesh.export(stl_file+'_filtered_surface.stl', file_type='stl_ascii')

# ############### Full model #################
viz_utils.visualizer.draw_mesh(mesh)

# ############################### PCA #################################

covariance_matrix = np.cov(mesh.vertices.T)
eigen_values, eigen_vectors = LA.eig(covariance_matrix)  # returns normalized eig vectors
idx = eigen_values.argsort()[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]


print('Gun model a, b', gun_model.a, gun_model.b)
ori_start = np.min(mesh.vertices, axis=0) # - eigen_vectors[:, use_eigen_vector_index] * (slicing_distance- gun_model.a) #  eigen_vectors[:, 2] * (eigen_values[2] * 10) +
stop = np.max(mesh.vertices, axis=0) # + eigen_vectors[:, use_eigen_vector_index] * slicing_distance
if use_eigen_vector_index == 0:
    ori_start[2] = stop[2] = 0
elif use_eigen_vector_index == 1:
    ori_start[0] = stop[0] = 0
    ori_start[1] = stop[1] = 0
elif use_eigen_vector_index == 2:
    ori_start[1] = stop[1] = 0

slice_direction = stop - ori_start
length = LA.norm(slice_direction)
slice_direction /= length
start = ori_start + slice_direction*starting_slice_offset
slice_direction = stop - start
length = LA.norm(slice_direction)
slice_direction /= length

if starting_slice_offset:
    viz_utils.plot_path(viz_utils.visualizer.axs_mesh, [ori_start ,start], color='yellow')

# print('start ', start, stop, length, np.arange(0, length, step=slicing_distance))
# print("Eigenvector: \n", eigen_vectors, "\n")

print('Selected EV', eigen_vectors[:, use_eigen_vector_index])
print('slice_direction', slice_direction)
viz_utils.plot_normals(viz_utils.visualizer.axs_mesh, [start], [eigen_vectors[:, use_eigen_vector_index]], norm_length=length, color='r', lw=1)
viz_utils.plot_normals(viz_utils.visualizer.axs_slice, [start], [slice_direction], norm_length=length, color='b', lw=1)
# print('eigen_vectors[:,0]', eigen_vectors[:, 0])

# ################################# Slicing #####################################

sections = mesh.section_multiplane(plane_origin=start,
                                   plane_normal=slice_direction,
                                   heights=np.arange(0, length, step=slicing_distance))

"""sections, to_3D, face_indexes = trimesh.intersections.mesh_multiplane(mesh, plane_origin=start,
                                   plane_normal=slice_direction,
                                   heights=np.arange(0, length, step=slicing_distance))"""
print('empty slices', [i for i, s in enumerate(sections) if not s])

print('sections', len(sections))
sections = [s for s in sections if s]
print('sections', len(sections))


d3sections = [section.to_3D() for section in sections]

face_indices = [path.metadata['face_index'] for path in sections]
face_normals = [mesh.face_normals[segment_face_indices] for segment_face_indices in face_indices]

# ############################ Trajectory generation ##############################

trajectory_generator = TrajectoryGenerator(mesh=mesh, gun_model=gun_model, standoff_dist=standoff_dist)
all_tool_locations, all_tool_normals, section_end_vert_pairs = trajectory_generator.generate_trajectory(d3sections)

# all_tool_locations = [ [v0 v1 v2 v3 .. vn ] , [v0 v1 v2 v3 .. vm] .. ]
# tool must be ON and move through [v0 v1 v2 v3 .. vn] continuously,
# and off between ..vn of one group and v0.. of the next
# Vert groups may or may not be from the same section

for i, (all_verts_this_section, tool_normals_this_section) in enumerate(zip(all_tool_locations, all_tool_normals)):
    plot_path(viz_utils.visualizer.axs_init, vertices=all_verts_this_section)
    viz_utils.plot_normals(viz_utils.visualizer.axs_init, vertices=all_verts_this_section,
                           directions=tool_normals_this_section)
    """
    if i > 0:
        plot_path(viz_utils.visualizer.axs_init, vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]],
                  color='k')
        plot_path(viz_utils.visualizer.final_path_ax, vertices=[all_verts_this_section[i - 1][-1], all_verts_this_section[i][0]],
                  color='k')
        vert_iter += 1
    """
for i in range(int(len(section_end_vert_pairs) / 2)):
    plot_path(viz_utils.visualizer.axs_init, vertices=[section_end_vert_pairs[i * 2], section_end_vert_pairs[i * 2 + 1]], color='k')

plt.draw()
plt.pause(0.001)

print('all_tool_locations\n', all_tool_locations, '\nall_tool_normals\n', all_tool_normals)



# Sample points on surface for simulation
samples, sample_face_indexes = trimesh.sample.sample_surface_even(mesh, number_of_samples, radius=None)
deposition_thickness = [0.0] * len(samples)
sample_tree = spatial.KDTree(samples)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(samples)
o3d.io.write_point_cloud("wall_surface.pcd", pcd)

# ################################# Calculate paint passes #################################
deposition_thickness = np.array(deposition_thickness)
scatter = viz_utils.visualizer.final_rendering_ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=surface_sample_viz_size, picker = 2)#, c=deposition_thickness, cmap='coolwarm')


# ############### Writing to a JSON file ##################
file_data = []
ray_mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
sample_dist = constant_vel * tool_motion_time_resolution
all_tool_positions, tool_normals = interpolate_tool_motion(all_tool_locations, all_tool_normals, sample_dist)
total_time_count = 0
for continuous_tool_positions, continuous_tool_normals in zip(all_tool_positions, tool_normals):
    time_stamp = 0

    plot_path(viz_utils.visualizer.final_path_ax, continuous_tool_positions)
    viz_utils.plot_normals(viz_utils.visualizer.final_path_ax, continuous_tool_positions, continuous_tool_normals)

    for pos_index, (current_tool_position, current_tool_normal) in enumerate(
            zip(continuous_tool_positions, continuous_tool_normals)):

        intersection_location, surface_normal = get_intersection_point(current_tool_position,  current_tool_normal,
                                                                       mesh, ray_mesh_intersector, sample_tree)

        current_tool_position, current_tool_normal = limit_tool_position(current_tool_position, intersection_location,
                                                                         current_tool_normal, tool_limits)

        if pos_index < len(continuous_tool_positions) - 1:
            next_intersection_location, next_surface_normal = get_intersection_point(
                continuous_tool_positions[pos_index + 1], continuous_tool_normals[pos_index + 1],
                mesh, ray_mesh_intersector, sample_tree)

            current_tool_travel_vec = next_intersection_location - intersection_location
            current_tool_travel_vec /= LA.norm(current_tool_travel_vec)

            travel_component_on_normal = np.dot(current_tool_normal, current_tool_travel_vec)
            current_tool_minor_axis_vec = current_tool_travel_vec - travel_component_on_normal*current_tool_normal
            current_tool_minor_axis_vec /= LA.norm(current_tool_minor_axis_vec)

            current_tool_major_axis_vec = np.cross(current_tool_minor_axis_vec, current_tool_normal)

        tool_pos_to_point = current_tool_position-intersection_location
        actual_norm_dist = LA.norm(tool_pos_to_point)
        tool_pos_to_point /= actual_norm_dist

        # Validate orthogonality of the 3 vectors
        print('(np.array(current_tool_normal), current_tool_major_axis_vec, current_tool_minor_axis_vec)\n', np.array(current_tool_normal), current_tool_major_axis_vec, current_tool_minor_axis_vec)
        viz_utils.plot_normals(viz_utils.visualizer.axs_temp, [current_tool_position], [current_tool_normal],
                               color='r', lw=1, hw=0.1)
        #viz_utils.plot_normals(viz_utils.visualizer.axs_temp, [current_tool_position], [current_tool_minor_axis_vec], color='g', lw=1, hw=0.1)
        # viz_utils.plot_normals(viz_utils.visualizer.axs_temp, [current_tool_position], [current_tool_major_axis_vec],lw=1,
        #                       color='b')

        # A = np.column_stack((np.array(current_tool_normal), current_tool_major_axis_vec, current_tool_minor_axis_vec))
        # I = A.T*A

        mag = LA.norm(np.dot(current_tool_normal, current_tool_minor_axis_vec))+LA.norm(np.dot(current_tool_minor_axis_vec, current_tool_major_axis_vec))

        time_scale = 1.0
        if tool_pitch_speed_compensation:
            time_scale = 1.0 / surface_scaling( gun_model.h, actual_norm_dist, surface_normal, tool_pos_to_point,
                                               current_tool_normal)
        dict = {"time_stamp": time_stamp*actual_norm_dist/standoff_dist,
                "tool_position": list(current_tool_position),
                "minor_axis_vec": current_tool_minor_axis_vec.tolist(),
                "major_axis_vec": current_tool_major_axis_vec.tolist(),
                "spray_on": False if i==len(continuous_tool_positions)-1 else True,
                "tool_normal": list(current_tool_normal),
                }
        file_data.append(dict)
        time_stamp += tool_motion_time_resolution
    total_time_count += time_stamp
with open('tool_positions.json', 'w') as outfile:
    json.dump(file_data, outfile, indent=2)


# ########### Create pose list from constant time interval ##############

sim_sample_dist = constant_vel * deposition_sim_time_resolution
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
        deposition_sim_time_resolution, gun_model, tool_limits

    if paint_pass >= len(total_tool_positions):
        print('\ndeposition_thickness\nmin:', deposition_thickness.min() * 1000, 'mm\nmax',
              deposition_thickness.max() * 1000, 'mm\ndiff: ', (deposition_thickness.max()-deposition_thickness.min())*1000,
              'mm\nstd:', deposition_thickness.std(0) * 1000, '\nmean:', deposition_thickness.mean(0) * 1000,
              '\nmedian:', np.median(deposition_thickness)*1000)
        #sns.distplot(deposition_thickness*1000, ax=viz_utils.visualizer.ax_distrib_hist)
        # plt.draw()
        #plt.show()
        animation.event_source.stop()
        animation.save_count = frame_number
    else:
        if j==0:
            continuous_tool_positions, continuous_tool_normals = all_tool_positions[paint_pass], tool_normals[paint_pass]
            # plot_path(viz_utils.visualizer.final_rendering_ax, continuous_tool_positions)
            # viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, continuous_tool_positions, continuous_tool_normals)
            tool_major_axis_vecs, tool_minor_axis_vecs = [], []
            # print('Estim axis vecs')
            current_tool_minor_axis_vec=[]
            current_tool_major_axis_vec=[]
            for pos_index, (current_tool_position, current_tool_normal) in enumerate(
                    zip(continuous_tool_positions, continuous_tool_normals)):
                # set minor axis direction to direction of movement
                intersection_location, surface_normal = get_intersection_point(
                    continuous_tool_positions[pos_index], continuous_tool_normals[pos_index],
                    mesh, ray_mesh_intersector, sample_tree)
                if pos_index < len(continuous_tool_positions)-1:
                    # if angle_between_vectors(current_tool_normal, continuous_tool_normals[pos_index+1]) > 0:
                    next_intersection_location, next_surface_normal = get_intersection_point(
                        continuous_tool_positions[pos_index + 1], continuous_tool_normals[pos_index + 1],
                        mesh, ray_mesh_intersector, sample_tree)

                    current_tool_travel_vec = next_intersection_location - intersection_location
                    current_tool_travel_vec /= LA.norm(current_tool_travel_vec)

                    travel_component_on_normal = np.dot(current_tool_normal, current_tool_travel_vec)

                    current_tool_minor_axis_vec = current_tool_travel_vec-travel_component_on_normal
                    current_tool_minor_axis_vec /= LA.norm(current_tool_minor_axis_vec)

                    current_tool_major_axis_vec = np.cross(current_tool_minor_axis_vec, current_tool_normal)
                tool_major_axis_vecs.append(current_tool_major_axis_vec)
                tool_minor_axis_vecs.append(current_tool_minor_axis_vec)

            #viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, continuous_tool_positions, tool_minor_axis_vecs, norm_length=0.3, color='g')

        intersection_location, surface_normal = get_intersection_point(
            continuous_tool_positions[j], continuous_tool_normals[j],mesh, ray_mesh_intersector, sample_tree)
        continuous_tool_positions[j], continuous_tool_normals[j] = limit_tool_position(continuous_tool_positions[j], intersection_location,
                                                                         continuous_tool_normals[j], tool_limits)

        tool_pos_to_point = continuous_tool_positions[j] - intersection_location
        actual_norm_dist = LA.norm(tool_pos_to_point)
        # print('intersection_index_tri', intersection_index_tri)
        # print('intersection_index_tri[j]', intersection_index_tri[j])

        time_scale = 1.0
        if tool_pitch_speed_compensation:
            time_scale = 1.0/ surface_scaling(gun_model.h, actual_norm_dist, surface_normal, tool_pos_to_point, continuous_tool_normals[j])

        if j%5==0:
            # viz_utils.visualizer.final_rendering_ax.scatter(intersection_location[0], intersection_location[1], intersection_location[2],
            #                            s=surface_sample_viz_size,
            #                            picker=2, c='r')  # , c=deposition_thickness, cmap='coolwarm')

            print(f'time_scale {time_scale: .3f}, actual_norm_dist {actual_norm_dist: .3f}, ')
            viz_utils.visualizer.ax_distrib_hist.cla()
            # viz_utils.visualizer.ax_distrib_hist.hist(deposition_thickness, color='blue', edgecolor='black', bins='auto', density=False)
            viz_utils.visualizer.ax_distrib_hist.set_xlabel('deposition thickness (mm)')
            binwidth = 0.01
            min_val,max_val = np.min(deposition_thickness)*1000, np.max(deposition_thickness)*1000
            val_width = (max_val - min_val)
            n_bins = int(val_width / binwidth)
            if n_bins==0:
                n_bins=1
            #print('bins', n_bins, val_width)
            sns.histplot(deposition_thickness * 1000, kde=True, bins=n_bins, ax=viz_utils.visualizer.ax_distrib_hist) # , binrange=(min_val, max_val)
            arange =  np.arange(min_val , max_val , binwidth)
            # print('np.arange(min_val , max_val , binwidth)', arange, 'max', max_val)
            # viz_utils.visualizer.ax_distrib_hist.set_xticks(np.arange(min_val -binwidth/2, max_val +binwidth/2, binwidth))
            #if arange.shape[0] > 0:
            #     viz_utils.visualizer.ax_distrib_hist.set_xlim(0, arange[-1]+binwidth/2)
            plt.draw()
        # time_scale = actual_norm_dist/standoff_dist
        # print('time_scale', time_scale, 'actual_norm_dist', actual_norm_dist, 'standoff_dist', standoff_dist)
        animation.event_source.interval = deposition_sim_time_resolution*1000*time_scale
        gun_model.set_h(actual_norm_dist)
        affected_points_for_tool_position(deposition_thickness, sample_tree, mesh,
                                           surface_normal, intersection_location,
                                           continuous_tool_positions[j], continuous_tool_normals[j],
                                           tool_major_axis_vecs[j], tool_minor_axis_vecs[j],
                                           gun_model, deposition_sim_time_resolution*time_scale, scatter)

        j += 1

        if j >= len(continuous_tool_positions):
            paint_pass +=1
            j=0

    return scatter,

# scatter.set_clim(vmin=min(deposition_thickness), vmax=max(deposition_thickness))
# scatter.set_color()

def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=20, n_circles_latitude=None):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z



# final_rendering_fig.canvas.set_window_title('Paint sim')

animation = FuncAnimation(viz_utils.visualizer.final_rendering_fig, update, interval= deposition_sim_time_resolution*1000, blit=False,
                          save_count=350, fargs=(scatter, deposition_thickness)) # , cache_frame_data=False, repeat = False)


# Writer = matplotlib.animation.writers['html']
# writer = Writer(fps=15, metadata={'artist':'COBOD'}, bitrate=1800)
# animation.save('paint_simulation.html', writer=writer)
#               #  progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))

"""

viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [all_tool_locations[0][0]], [all_tool_normals[0][0]])
intersection_location, surface_normal = get_intersection_point(all_tool_locations[0][0], all_tool_normals[0][0],
                                                                       mesh, ray_mesh_intersector, sample_tree)
viz_utils.visualizer.final_rendering_ax.scatter(*all_tool_locations[0][0], s=40, c='g')
viz_utils.visualizer.final_rendering_ax.scatter(intersection_location[0], intersection_location[1], intersection_location[2], s=40, c='r')
frame_xs, frame_ys, frame_zs = WireframeSphere(centre = intersection_location, radius=gun_model.a)
viz_utils.visualizer.final_rendering_ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.1)



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

"""
plt.show()
print('after plot')
