from spray_gun_model import *

gun_model2 = SprayGunModel()
gun_model1 = SprayGunModel()

def cost_function(overlap_dist, z_orientation_1, z_orientation_2):
    gun_model1.z_orientation = z_orientation_1
    gun_model2.z_orientation = z_orientation_2

    canvas_1 = np.array(gun_model1.get_deposition_canvas(z_orientation_1))
    canvas_2 = np.array(gun_model2.get_deposition_canvas(z_orientation_2))

    half_canvas_1 = canvas_1[:, int(len(canvas_1[0]) / 2):]
    half_canvas_2 = canvas_2[:, :int(len(canvas_2[0]) / 2)]

    no_bins_1 = int(gun_model1.resolution * overlap_dist / 2)
    no_bins_2 = int(gun_model2.resolution * overlap_dist / 2)

    full_canvas_1_padded = np.pad(half_canvas_1, (0 ,no_bins_2), 'constant', constant_values = (0, 0))
    full_canvas_2_padded = np.pad(half_canvas_2, (no_bins_1, 0), 'constant', constant_values=(0, 0))

    gun_model1.visualize_deposition(full_canvas_1_padded)
    gun_model2.visualize_deposition(full_canvas_2_padded)
    print('full_canvas_1_padded', full_canvas_1_padded.shape)
    print('full_canvas_2_padded', full_canvas_2_padded.shape)

if __name__ == '__main__':
    cost_function(0.1, np.radians(0), np.radians(45))