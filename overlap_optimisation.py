from spray_gun_model import *
from scipy.optimize import least_squares


def get_1d_overlap_profile(gun_model, overlap_dist, z_orientation_1, z_orientation_2, visualize = False):
    # Assuming resolutions are same
    g1_profile, g1_x_locations = gun_model.get_half_1d_profile(z_orientation_1)
    g2_profile, g2_x_locations = gun_model.get_half_1d_profile(z_orientation_2)
    np.nan_to_num(g1_profile, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(g2_profile, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Padding
    # print('overlap_dist',overlap_dist)
    padding = int(overlap_dist/gun_model.resolution)+1
    g1_profile = np.pad(g1_profile, (0, padding), 'constant', constant_values=(0, 0))

    g2_profile = np.pad(g2_profile, (0, padding), 'constant', constant_values=(0, 0))
    g2_profile = np.flip(g2_profile)
    x_locations = np.arange(0,  g1_profile.shape[0], step = 1)*gun_model.resolution
    if visualize:
        plt.plot(x_locations, g1_profile,'r')
        plt.plot(x_locations, g2_profile, 'g')
        plt.plot(x_locations, g1_profile+g2_profile, 'b')
        plt.show()
    return g1_profile+g2_profile, x_locations


def cost_function(input_1d_profile) -> float:
    return max(input_1d_profile) - min(input_1d_profile)


def opt_wrapper(overlap_dist, gun_model, z_orientation_1, z_orientation_2 ):
    profile, x_locations = get_1d_overlap_profile(gun_model, overlap_dist, z_orientation_1, z_orientation_2)
    return cost_function(profile)


def get_optimal_overlap_distance(gun_model, z_orientation_1, z_orientation_2) -> float:
    # canvas, X_grid, Y_grid = gun_model.get_deposition_canvas(z_orientation_1)
    # gun_model.visualize_deposition(canvas, X_grid, Y_grid)
    x = np.array([0.1])
    result = least_squares(opt_wrapper, x, args=(gun_model, z_orientation_1, z_orientation_2), gtol=None, diff_step= gun_model.resolution)
    print('result', result)
    print('values', result.x)

    return result.x[0]


def get_overlap_profile(gun_model, overlap_dist, z_orientation_1, z_orientation_2):
    gun_model.z_orientation = z_orientation_1
    gun_model.z_orientation = z_orientation_2

    # Resolution should be equal
    min_res = gun_model.resolution

    canvas_1, X_grid_1, Y_grid_1 = gun_model.get_deposition_canvas(z_orientation_1)
    canvas_2, X_grid_2, Y_grid_2 = gun_model.get_deposition_canvas(z_orientation_2)

    # Find the index of the closest point to overlap dist
    c1_x_index = (np.abs(X_grid_1[0] - overlap_dist / 2)).argmin()
    c2_x_index = (np.abs(X_grid_2[0] - overlap_dist / 2)).argmin()

    # Find maximum height of the two canvases
    max_height_bins = max(canvas_1.shape[0], canvas_2.shape[0])
    max_half_height = max(Y_grid_1[:, 0], Y_grid_2[:, 0])
    new_y_arr = np.arange(-max_half_height, max_half_height, min_res)

    # Vertical padding
    if canvas_1.shape[0]<max_height_bins:
        vertical_diff = int((max_height_bins-canvas_1.shape[0])/2)
        canvas_1_padded = np.pad(canvas_1, (vertical_diff, vertical_diff), 'constant', constant_values=(0, 0))

    if canvas_2.shape[0] < max_height_bins:
        vertical_diff = int((max_height_bins - canvas_2.shape[0]) / 2)
        canvas_2_padded = np.pad(canvas_2, (vertical_diff, vertical_diff), 'constant', constant_values=(0, 0))

    # Horizontal padding
    canvas_1_padded = np.pad(canvas_1, (vertical_diff, vertical_diff), 'constant', constant_values=(0, 0))

    right_half_canvas_1 = canvas_1_padded[:, c1_x_index:]
    left_half_canvas_2 = canvas_2_padded[:, :c2_x_index]

    no_bins_1 = int(min_res * overlap_dist / 2)
    no_bins_2 = int(min_res * overlap_dist / 2)

    # shape = [max(a.shape[axis] for a in (g1_profile, g1_profile)) for axis in range(len(g1_profile.shape))]
    full_canvas_1_padded = np.pad(right_half_canvas_1, (0, no_bins_2), 'constant', constant_values=(0, 0))
    full_canvas_2_padded = np.pad(left_half_canvas_2, (no_bins_1, 0), 'constant', constant_values=(0, 0))

    # gun_model1.visualize_deposition(full_canvas_1_padded)
    gun_model.visualize_deposition(canvas_2)
    print('full_canvas_1_padded', full_canvas_1_padded.shape)
    print('full_canvas_2_padded', full_canvas_2_padded.shape)


if __name__ == '__main__':
    gun_model = SprayGunModel()
    d = get_optimal_overlap_distance(gun_model, np.radians(0), 0)
    get_1d_overlap_profile(gun_model, d, 0, 0, True)
    get_overlap_profile(gun_model, d, 0, 0)

