import numpy as np
from numpy import linalg as LA
import matplotlib
import viz_utils

def affected_points_for_tool_position(deposition_thickness, sample_tree,sample_face_indexes, mesh,
                                       intersection_location,
                                       tool_position, tool_normal,
                                       tool_major_axis_vec, tool_minor_axis_vec,
                                       gun_model, deposition_sim_time_resolution, scatter):

    # Take the larger axis and double it = search radius
    query_ball_points = sample_tree.query_ball_point(intersection_location,
                                                     (gun_model.b if gun_model.a <= gun_model.b else gun_model.a) * 4.0)
    k = 0
    for point_index in query_ball_points:
        point = sample_tree.data[point_index]
        surface_normal = mesh.face_normals[sample_face_indexes[point_index]]
        # print(k, end=' ')
        k += 1
        normal_dist_actual = LA.norm(intersection_location - tool_position)
        tool_pos_to_point = point - tool_position
        tool_pos_to_point_unit = tool_pos_to_point/LA.norm(tool_pos_to_point)
        normal_dist_h_dash = LA.norm(np.dot(tool_pos_to_point, tool_normal))
        # print('normal_dist_actual/normal_dist_h_dash', normal_dist_actual/normal_dist_h_dash, 'normal_dist_actual', normal_dist_actual, 'normal_dist_h_dash', normal_dist_h_dash)
        tool_pos_to_proj_point = tool_pos_to_point_unit*(LA.norm(tool_pos_to_point)*normal_dist_actual/normal_dist_h_dash)
        rp = tool_pos_to_proj_point - tool_normal * normal_dist_actual

        # Debugging
        #viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [tool_position], [tool_normal], color='g', lw=1, norm_length=gun_model.h)
        #viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [tool_position + tool_normal*gun_model.h], [rp/LA.norm(rp)], color='b', lw=1, norm_length=LA.norm(rp))
        #viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [tool_position], [tool_pos_to_point_unit], color='r', lw=1, norm_length = LA.norm(tool_pos_to_proj_point))
        #viz_utils.plot_normals(viz_utils.visualizer.final_rendering_ax, [point], [surface_normal], color='r', lw=1, norm_length=0.1)

        x, y = np.dot(rp, tool_major_axis_vec), np.dot(rp, tool_minor_axis_vec)

        if gun_model.check_point_validity(x, y):
            # Estimate deposition thickness for this point
            deposition_at_h_dash = gun_model.deposition_intensity(x, y)
            tool_pos_to_point_dist = LA.norm(tool_pos_to_point)
            # multiplier = ((gun_model.h/tool_pos_to_point_dist)**2)*np.dot(tool_pos_to_point_unit, -surface_normal)/np.dot(tool_pos_to_point_unit, tool_normal)**3
            # multiplier = ((gun_model.h/normal_dist_h_dash)**2)*np.dot(-tool_normal, surface_normal)
            multiplier = ((LA.norm(tool_pos_to_proj_point)/tool_pos_to_point_dist)**2)*(np.dot(-tool_normal, surface_normal))
            # multiplier = 1
            if multiplier<0.02:
                print('multiplier', multiplier)
            deposition_thickness[point_index] += multiplier * deposition_at_h_dash * deposition_sim_time_resolution
    # print('\nDrawing', max(deposition_thickness))
    n = matplotlib.colors.Normalize(vmin=min(deposition_thickness),
                                    vmax=max(deposition_thickness))
    m = matplotlib.cm.ScalarMappable(norm=n, cmap='YlOrBr_r')
    scatter.set_color(m.to_rgba(deposition_thickness))
    scatter._facecolor3d = scatter.get_facecolor()
    scatter._edgecolor3d = scatter.get_edgecolor()
    # print('\ndone')

