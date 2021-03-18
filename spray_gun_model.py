import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class SprayGunModel:
    def __init__(self, beta1=1.5, beta2=2.5, a=0.3, b=0.1, f_max=0.0001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        self.b = b
        self.f_max = f_max
        self.resolution = 0.001
        self.viz_resolution = 0.01
        self.z_orientation = 0
        self.h = 0.5

    def check_point_validity(self, x, y) -> bool:
        # (self.b * np.sqrt(1 - (x ** 2) / self.a ** 2))
        return abs(x) < self.a and abs(y) < self.b

    def deposition_intensity(self, x, y) -> float:
        intensity = np.clip(self.f_max * pow(1.0 - (x ** 2) / self.a ** 2, self.beta1 - 1) * pow(
            1.0 - (y ** 2) / ((self.b ** 2) * (1 - (x ** 2) / self.a ** 2)), self.beta2 - 1), 0, self.f_max)
        intensity = np.nan_to_num(intensity, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        return intensity

    # Used for finding optimal overlap distance
    def get_half_1d_profile(self, orientation: float) -> (object, object):
        profile = []
        x_locations = []
        for x_ in np.arange(0, max(self.a, self.b), self.resolution):
            rotated_x = x_ * np.cos(orientation)
            rotated_y = x_ * np.sin(orientation)
            if self.check_point_validity(rotated_x, rotated_y):
                profile.append(self.deposition_intensity(rotated_x, rotated_y))
                x_locations.append(x_)
        return np.array(profile), np.array(x_locations)

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

    def visualize_deposition(self, template, X_grid, Y_grid):

        fig = plt.figure(figsize=(8, 3))
        fig.tight_layout()
        fig.canvas.set_window_title('Surface deposition intensity')
        fig.subplots_adjust(left=0.05, right=0.95, top=1.3, bottom=-0.2)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(X_grid, Y_grid, template,
                         antialiased=False, cmap="coolwarm", lw=0.5, rstride=1, cstride=1, alpha=0.5)
        ax2.contour(X_grid, Y_grid, template, 10, lw=3, colors="k", linestyles="solid")
        # ax.contour(X_grid, Y_grid, template, zdir='z', offset=self.f_max*1.5, cmap="coolwarm")
        ax2.contour(X_grid, Y_grid, template, zdir='x', offset=np.min(X_grid[0]), cmap="coolwarm")
        ax2.contour(X_grid, Y_grid, template, zdir='y', offset=np.max(Y_grid[:, 0]), cmap="coolwarm")

        ax1.imshow(template, extent=[np.min(X_grid[0]), np.max(X_grid[0]), np.min(Y_grid[:, 0]), np.max(Y_grid[:, 0])])

        # limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        # ax.set_box_aspect(np.ptp(limits, axis=1))

        """min_x, max_x = np.min(X_grid[0]), np.max(X_grid[0])
        min_y, max_y = np.min(Y_grid[0]), np.max(Y_grid[0])
        min_z, max_z = np.min(np.min(template, axis = 0)), np.max(np.max(template, axis = 0))

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_z, max_y)
        ax.set_zlim(min_z, max_z)"""

        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        # limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        # ax.set_box_aspect(np.ptp(limits, axis=1))
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # ax.set_xlim3d(-max(self.a, self.b), max(self.a, self.b))
        # ax.set_ylim3d(-max(self.a, self.b), max(self.a, self.b))

        """min_x, max_x = np.min(X_grid[0]), np.max(X_grid[0])
        min_y, max_y = np.min(Y_grid[0]), np.max(Y_grid[0])
        min_z, max_z = np.min(np.min(template, axis = 0)), np.max(np.max(template, axis = 0))

        max_range = np.array([max_x - min_x, max_y - min_y, max_z - min_z]).max() / 2.0

        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        mid_z = (max_z + min_z) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)"""
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    gun_model = SprayGunModel()

    canvas, X_grid, Y_grid = gun_model.get_deposition_canvas(np.radians(0))
    gun_model.visualize_deposition(canvas, X_grid, Y_grid)

    prof, locations = gun_model.get_half_1d_profile(np.radians(0))
    print('get_half_1d_profile', prof)
    # fig = plt.figure()
    # plt.plot(locations, prof)
    plt.show()
