
from spray_gun_model import *
from scipy.optimize import least_squares
from spray_gun_model import SprayGunModel


def get_1d_overlap_profile(gun_model, overlap_dist, z_orientation_1, z_orientation_2, visualize = False):

    if overlap_dist<0:
        print('d is < 0: ', overlap_dist, 'Optimisation will fail')
    # Assuming resolutions are same
    g1_profile, g1_x_locations = gun_model.get_half_1d_profile(z_orientation_1)
    g2_profile, g2_x_locations = gun_model.get_half_1d_profile(z_orientation_2)
    np.nan_to_num(g1_profile, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(g2_profile, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Padding
    padding = int(overlap_dist/gun_model.sim_resolution)+1

    g1_profile = np.pad(g1_profile, (0, padding), 'constant', constant_values=(0, 0))
    g2_profile = np.pad(g2_profile, (0, padding), 'constant', constant_values=(0, 0))

    # Flip right-side half profile
    g2_profile = np.flip(g2_profile)
    x_locations = np.arange(0,  g1_profile.shape[0], step = 1)*gun_model.sim_resolution
    final_profile = g1_profile+g2_profile
    if visualize:
        fig, ax = plt.subplots(figsize=(5,3))
        fig.subplots_adjust(bottom=0.164)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Deposition (m)')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-5,-3))
        actual_seperation_dist = float(overlap_dist)+gun_model.a
        fig.canvas.set_window_title(f'Overlap optimization d={overlap_dist: .2f}')
        ax.plot(x_locations, g1_profile,'r', linestyle='-.')
        ax.plot(x_locations, g2_profile, 'g', linestyle='-.')
        ax.plot(x_locations, final_profile, 'b')
        plt.axhline(y=np.max(final_profile[1:-1]), color='b', linestyle='dotted')
        plt.axhline(y=np.min(final_profile[1:-1]), color='b', linestyle='dotted')

        plt.draw()
        plt.pause(0.001)
        # plt.show()
    return final_profile, x_locations


def cost_function(input_1d_profile) -> float:
    return max(input_1d_profile[1:-1]) - min(input_1d_profile[1:-1])


def opt_wrapper(overlap_dist, gun_model, z_orientation_1, z_orientation_2 ):
    profile, x_locations = get_1d_overlap_profile(gun_model, overlap_dist, z_orientation_1, z_orientation_2)
    return cost_function(profile)


def get_optimal_overlap_distance(gun_model, z_orientation_1, z_orientation_2) -> float:
    x = np.array([0.1])
    result = least_squares(opt_wrapper, x, args=(gun_model, z_orientation_1, z_orientation_2), gtol=None, diff_step= gun_model.sim_resolution)

    print('result', result)
    print('values', result.x)

    return result.x[0]


def disp_overlap_profile(gun_model, overlap_dist, z_orientation_1, z_orientation_2):
    gun_model.z_orientation = z_orientation_1
    gun_model.z_orientation = z_orientation_2

    # Resolution should be equal
    min_res = gun_model.viz_resolution

    canvas_1, X_grid_1, Y_grid_1 = gun_model.get_deposition_canvas(z_orientation_1)
    canvas_2, X_grid_2, Y_grid_2 = gun_model.get_deposition_canvas(z_orientation_2)

    # Find maximum height of the two canvases
    max_height_bins = max(canvas_1.shape[0], canvas_2.shape[0])

    # Vertical padding
    if canvas_1.shape[0]<max_height_bins:
        vertical_diff = int((max_height_bins-canvas_1.shape[0])/2)
        canvas_1 = np.pad(canvas_1, (vertical_diff, vertical_diff), 'constant', constant_values=(0, 0))

    if canvas_2.shape[0] < max_height_bins:
        vertical_diff = int((max_height_bins - canvas_2.shape[0]) / 2)
        canvas_2 = np.pad(canvas_2, (vertical_diff, vertical_diff), 'constant', constant_values=(0, 0))

    # Horizontal padding
    no_bins_1 = int((overlap_dist+gun_model.a)/min_res)
    no_bins_2 = int((overlap_dist+gun_model.a)/min_res)

    full_canvas_1_padded = np.pad(canvas_1, pad_width=((0,0), (0, no_bins_1)), mode='constant', constant_values=0)
    full_canvas_2_padded = np.pad(canvas_2, pad_width=((0,0), (no_bins_2, 0)), mode='constant', constant_values=0)
    combined = np.maximum(full_canvas_1_padded, full_canvas_2_padded)

    new_y_arr = (np.arange(0, full_canvas_1_padded.shape[0], 1) - full_canvas_1_padded.shape[0]/2)*min_res
    new_x_arr = (np.arange(0, full_canvas_1_padded.shape[1], 1) - full_canvas_1_padded.shape[1]/2)*min_res


if __name__ == '__main__':
    gun_model = SprayGunModel()
    canvas, X_grid, Y_grid = gun_model.get_deposition_canvas(np.radians(0))
    prof, locations = gun_model.get_half_1d_profile(np.radians(0))
    fig = plt.figure()
    plt.plot(locations, prof)
    viz_utils.visualize_deposition(canvas, X_grid, Y_grid)

    d = 0.01
    try:
        d = get_optimal_overlap_distance(gun_model, np.radians(0), 0)
    except:
        print('Opt FAILED')
    print('Optimal distance', d)
    print('a, b', gun_model.a, gun_model.b)

    costs = []
    d_locs = np.arange(0, gun_model.a, 0.02)
    for d_i in d_locs:
        final_profile, x_locations = get_1d_overlap_profile(gun_model, d_i, 0, 0, False)
        costs.append(cost_function(final_profile))

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.subplots_adjust(bottom=0.164)
    ax.set_xlabel('d (m)')
    ax.set_ylabel('cost')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-5, -3))
    fig.canvas.set_window_title('Cost function')
    ax.plot(d_locs, np.array(costs))

    get_1d_overlap_profile(gun_model, 0, 0, 0, True)
    get_1d_overlap_profile(gun_model, d/2, 0, 0, True)
    get_1d_overlap_profile(gun_model, d, 0, 0, True)
    get_1d_overlap_profile(gun_model, gun_model.a/2, 0, 0, True)
    get_1d_overlap_profile(gun_model, gun_model.a, 0, 0, True)

    plt.show()

