import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import viz_utils


class SprayGunModel:
    def __init__(self, beta1=2.5, beta2=2.5, maj_axis_angle = np.radians(10), min_axis_angle = np.radians(5), f_max=0.0012):
        self.beta1 = beta1
        self.beta2 = beta2
        self.maj_axis_angle = maj_axis_angle
        self.min_axis_angle = min_axis_angle
        self.set_h(0.5)
        self.f_max = f_max
        self.sim_resolution = 0.001
        self.viz_resolution = 0.01
        self.z_orientation = 0

    def set_h(self, h: float):
        self.h = 0.5
        self.a = np.tan(self.maj_axis_angle) / self.h  # 1.0
        self.b = np.tan(self.min_axis_angle) / self.h  # 0.4


    def visualize_spray_cone(self):
        # TODO
        pass

    def check_point_validity(self, x, y) -> bool:
        # (self.b * np.sqrt(1 - (x ** 2) / self.a ** 2))
        return abs(x) < self.a and abs(y) < self.b

    def deposition_intensity(self, x, y) -> float:
        intensity=0
        try:
            intensity = np.clip(self.f_max * pow(1.0 - (x ** 2) / self.a ** 2, self.beta1 - 1) * pow(
                1.0 - (y ** 2) / ((self.b ** 2) * (1 - (x ** 2) / self.a ** 2)), self.beta2 - 1), 0, self.f_max)
            intensity = np.nan_to_num(intensity, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            pass
        return intensity

    # Used for finding optimal overlap distance
    def get_half_1d_profile(self, orientation: float) -> (object, object):
        profile = []
        x_locations = []
        for x_ in np.arange(0, max(self.a, self.b), self.sim_resolution):
            rotated_x = x_ * np.cos(orientation)
            rotated_y = x_ * np.sin(orientation)
            if self.check_point_validity(rotated_x, rotated_y):
                profile.append(self.deposition_intensity(rotated_x, rotated_y))
                x_locations.append(x_)
        return np.array(profile), np.array(x_locations)

    # TODO: show gun standoff in visualization
    # Mainly for visualization
    def get_deposition_canvas(self, orientation: float) -> object:
        c, s = np.cos(orientation), np.sin(orientation)
        R = np.array(((c, -s), (s, c)))
        a_rotated_point = np.absolute(R.dot(np.array([self.a, 0])))
        b_rotated_point = np.absolute(R.dot(np.array([0, self.b])))
        stacked = np.vstack([a_rotated_point, b_rotated_point])
        rotated_ab = np.max(stacked, axis=0)

        plot_arr_x = np.arange(-rotated_ab[0] - self.viz_resolution, rotated_ab[0] + self.viz_resolution,
                               self.viz_resolution)
        plot_arr_y = np.arange(-rotated_ab[1] - self.viz_resolution, rotated_ab[1] + self.viz_resolution,
                               self.viz_resolution)

        X_grid, Y_grid = np.meshgrid(plot_arr_x, plot_arr_y)
        deposition_template = np.zeros((len(plot_arr_y), len(plot_arr_x)))

        orientation = -orientation
        print('start_populating', deposition_template.shape, X_grid.shape, Y_grid.shape)
        for x_i in range(len(plot_arr_x)):
            for y_i in range(len(plot_arr_y)):
                x_, y_ = plot_arr_x[x_i], plot_arr_y[y_i]
                rotated_x = x_ * np.cos(orientation) - y_ * np.sin(orientation)
                rotated_y = x_ * np.sin(orientation) + y_ * np.cos(orientation)

                if self.check_point_validity(rotated_x, rotated_y):
                    deposition_template[y_i][x_i] = self.deposition_intensity(rotated_x, rotated_y)
                    # if abs(rotated_y)<0.001:
                    #    print(x_, deposition_template[y_i][x_i])
        np.nan_to_num(deposition_template, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        print('end_populating')
        return deposition_template, X_grid, Y_grid

    def __str__(self):
        return 'SprayGunModel('+str(vars(self))+')'


if __name__ == '__main__':
    gun_model = SprayGunModel()
    print('Gun Model', gun_model)
    canvas, X_grid, Y_grid = gun_model.get_deposition_canvas(np.radians(-45))
    viz_utils.visualize_deposition(canvas, X_grid, Y_grid)

    prof, locations = gun_model.get_half_1d_profile(np.radians(0))
    # fig = plt.figure()
    # plt.plot(locations, prof)
    plt.show()
