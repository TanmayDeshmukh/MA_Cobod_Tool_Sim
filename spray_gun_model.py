import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class SprayGunModel:
    def __init__(self, beta1=1.5, beta2=2.0, a=0.25, b=0.2, f_max=0.0001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        self.b = b
        self.f_max = f_max
        self.resolution = 0.001
        self.z_orientation = 0
        self.h = 0.2

    def check_point_validity(self, x, y) -> bool:
        # print('abs(x)', abs(x), abs(y))
        return abs(x) <= self.a and abs(y) <= self.b # (self.b * np.sqrt(1 - (x ** 2) / self.a ** 2))

    def deposition_intensity(self, x, y) -> float:
        # x_r = x * np.cos(orientation) - y * np.sin(orientation)
        # y_r = x * np.sin(orientation) + y * np.cos(orientation)
        # print(x**2, y**2,  )
        # if -self.a <=x <=self.a-self.b*np.sqrt(1-(x**2)/(self.a**2)) <=y <=self.b*np.sqrt(1-(x**2)/(self.a**2)):
        # if self.check_point_validity(x, y):
        # print('yes')
        return np.clip(self.f_max * pow(1.0 - (x ** 2) / self.a ** 2, self.beta1 - 1) * pow(
            1.0 - (y ** 2) / ((self.b ** 2) * (1 - (x ** 2) / self.a ** 2)), self.beta2 - 1), 0, self.f_max)
        # print('nope')
        # return 0.0

    def get_deposition_canvas(self, orientation: float) -> [[]]:

        # print('a', spray_gun_model.a, 'b', spray_gun_model.b)
        rotated_a = abs(self.a * np.cos(orientation) + self.b * np.sin(orientation))
        rotated_b = abs(self.a * np.sin(orientation) + self.b * np.cos(orientation))

        # rotated_a = max(spray_gun_model.a, spray_gun_model.b)
        # rotated_b = max(spray_gun_model.a, spray_gun_model.b)

        deposition_template = [[0] * int(rotated_a * 2 / self.resolution) for i in
                               range(int(rotated_b * 2 / self.resolution))]
        # print('rotated_a', rotated_a, 'rotated_b', rotated_b)
        # print(len(deposition_template), len(deposition_template[0]))
        print('start_populating', len(deposition_template[0]), len(deposition_template), rotated_a / self.resolution,
              rotated_b / self.resolution)
        for x_ in np.arange(-rotated_a, rotated_a, self.resolution):
            for y_ in np.arange(-rotated_b, rotated_b, self.resolution):
                index_x = int(x_ / self.resolution + len(deposition_template[0]) / 2)
                index_y = int(y_ / self.resolution + len(deposition_template) / 2)
                # print(index_x, index_y, len(deposition_template[0]))
                # try:
                x_r = x_ * np.cos(orientation) - y_ * np.sin(orientation)
                y_r = x_ * np.sin(orientation) + y_ * np.cos(orientation)
                # if abs(y_r) <= (spray_gun_model.b * np.sqrt(1 - (x_r ** 2) / spray_gun_model.a ** 2)):
                if 0 < index_x < len(deposition_template[0]) and 0 < index_y < len(deposition_template) and self.check_point_validity(x_r, y_r):
                    deposition_template[index_y][index_x] = self.deposition_intensity(x_r, y_r)
                    if deposition_template[index_y][index_x]<0:
                        print('xy', x_r, y_r, deposition_template[index_y][index_x])
                    # ax.scatter(x_, y_, deposition_template[index_x][index_y], marker=('o'))
                # except:
                #    print('tried', index_x, index_y)
        # plt.show()
        print('end_populating')
        return np.array(deposition_template)

    def visualize_deposition(self, template):

        print('min', template.min(axis=0).min(), 'max', template.max(axis=0).max())
        rotated_a = abs(self.a * np.cos(self.z_orientation) + self.b * np.sin(self.z_orientation))
        rotated_b = abs(self.a * np.sin(self.z_orientation) + self.b * np.cos(self.z_orientation))

        template_shape = template.shape
        plot_arr_x = np.arange(-(template_shape[1]*self.resolution/2), (template_shape[1]*self.resolution/2),
                               self.resolution)

        plot_arr_y = np.arange(-(template_shape[0]*self.resolution/2), (template_shape[0]*self.resolution/2),
                               self.resolution)
        # print('Shape', plot_arr_x.shape, plot_arr_y.shape, np_dep_temp.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(plot_arr_x, plot_arr_y)
        surf = ax.plot_surface(X, Y, template,
                               linewidth=0, antialiased=False)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlim3d(-max(self.a, self.b), max(self.a, self.b))
        ax.set_ylim3d(-max(self.a, self.b), max(self.a, self.b))
        plt.show()

if __name__ == '__main__':
    gun_model = SprayGunModel()

    canvas = gun_model.get_deposition_canvas(0)
    gun_model.visualize_deposition(canvas)