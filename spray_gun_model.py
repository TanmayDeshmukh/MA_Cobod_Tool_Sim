import numpy as np
class SprayGunModel:
    def __init__(self, beta1=1.5, beta2=4, a=0.1, b=0.2, f_max=1, resolution=0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        self.b = b
        self.f_max = f_max
        self.resolution = resolution
        self.z_orientation = 0

    def deposition_intensity(self, x, y) -> float:
        # x_r = x * np.cos(orientation) - y * np.sin(orientation)
        # y_r = x * np.sin(orientation) + y * np.cos(orientation)
        # print(x**2, y**2,  )
        if abs(y) <= (self.b * np.sqrt(1 - (x ** 2) / self.a ** 2)):
            return self.f_max * pow(1.0 - (x ** 2) / self.a ** 2, self.beta1 - 1) * pow(
                1.0 - (y ** 2) / ((self.b ** 2) * (1 - (x ** 2) / self.a ** 2)), self.beta2 - 1)
        # print('nope')
        return 0.0

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
                if 0 < index_x < len(deposition_template[0]) and 0 < index_y < len(deposition_template):
                    deposition_template[index_y][index_x] = self.deposition_intensity(x_r, y_r)
                    # ax.scatter(x_, y_, deposition_template[index_x][index_y], marker=('o'))
                # except:
                #    print('tried', index_x, index_y)
        # plt.show()
        print('end_populating')
        return deposition_template