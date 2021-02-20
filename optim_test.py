import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import stl
import time

resolution = 0.05
orientation = np.pi / 4.0


class SprayGunModel:
    def __init__(self, beta1=1.5, beta2=4, a=1, b=2, f_max=1):
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        self.b = b
        self.f_max = f_max

    def deposition_intensity(self, x, y):
        # x_r = x * np.cos(orientation) - y * np.sin(orientation)
        # y_r = x * np.sin(orientation) + y * np.cos(orientation)
        # print(x**2, y**2,  )
        if abs(y) <= (self.b * np.sqrt(1 - (x ** 2) / self.a ** 2)):
            return self.f_max * pow(1.0 - (x ** 2) / self.a ** 2, self.beta1 - 1) * pow(
                1.0 - (y ** 2) / ((self.b ** 2) * (1 - (x ** 2) / self.a ** 2)), self.beta2 - 1)
        # print('nope')
        return 0.0


spray_gun_model = SprayGunModel()
# deposition_template = np.zeros((int(spray_gun_model.a/resolution), int(spray_gun_model.b/resolution)))

# print('a', spray_gun_model.a, 'b', spray_gun_model.b)
rotated_a = abs(spray_gun_model.a * np.cos(orientation) + spray_gun_model.b * np.sin(orientation))
rotated_b = abs(spray_gun_model.a * np.sin(orientation) + spray_gun_model.b * np.cos(orientation))

# rotated_a = max(spray_gun_model.a, spray_gun_model.b)
# rotated_b = max(spray_gun_model.a, spray_gun_model.b)

deposition_template = [[0] * int(rotated_a * 2 / resolution) for i in
                       range(int(rotated_b * 2 / resolution))]
# print('rotated_a', rotated_a, 'rotated_b', rotated_b)
# print(len(deposition_template), len(deposition_template[0]))
start_time = time.time()
print('start_populating', len(deposition_template[0]), len(deposition_template), rotated_a / resolution,
      rotated_b / resolution)
for x_ in np.arange(-rotated_a, rotated_a, resolution):
    for y_ in np.arange(-rotated_b, rotated_b, resolution):
        index_x = int(x_ / resolution + len(deposition_template[0]) / 2)
        index_y = int(y_ / resolution + len(deposition_template) / 2)
        # print(index_x, index_y, len(deposition_template[0]))
        # try:
        x_r = x_ * np.cos(orientation) - y_ * np.sin(orientation)
        y_r = x_ * np.sin(orientation) + y_ * np.cos(orientation)
        # if abs(y_r) <= (spray_gun_model.b * np.sqrt(1 - (x_r ** 2) / spray_gun_model.a ** 2)):
        if 0 < index_x < len(deposition_template[0]) and 0 < index_y < len(deposition_template):
            deposition_template[index_y][index_x] = spray_gun_model.deposition_intensity(x_r, y_r)
            # ax.scatter(x_, y_, deposition_template[index_x][index_y], marker=('o'))
        # except:
        #    print('tried', index_x, index_y)
# plt.show()
print('end_populating')

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
np_dep_temp = np.zeros((int(rotated_b * 2 / resolution), int(rotated_a * 2 / resolution)))
print('start_populating2', len(deposition_template[0]), len(deposition_template), rotated_a / resolution,
      rotated_b / resolution)
x_len = int(len(deposition_template[0]) / 2)
y_len = int(len(deposition_template) / 2)
for index_x in range(-x_len, x_len):
    for index_y in range(-y_len, 0):
        x_ = index_x * resolution
        y_ = index_y * resolution
        x_r = x_ * np.cos(orientation) - y_ * np.sin(orientation)
        y_r = x_ * np.sin(orientation) + y_ * np.cos(orientation)
        np_dep_temp[index_y + y_len, index_x + x_len] = spray_gun_model.deposition_intensity(x_r, y_r)
        # ax.scatter(x_, y_, deposition_template[index_x][index_y], marker=('o'))
        # except:
        #    print('tried', index_x, index_y)
# plt.show()
np_dep_temp = np_dep_temp + np.flip(np.flip(np_dep_temp, 0), 1)
print('end_populating2')
# print(deposition_template)
# print('max', max(deposition_template))

print("--- %s seconds ---" % (time.time() - start_time))

# plt.imshow(np_dep_temp, cmap='hot')
# plt.show()


plot_arr_x = np.arange(-rotated_a, rotated_a - resolution,
                       resolution)

plot_arr_y = np.arange(-rotated_b, rotated_b - resolution,
                       resolution)
print('Shape', plot_arr_x.shape, plot_arr_y.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(plot_arr_x, plot_arr_y)
surf = ax.plot_surface(X, Y, np_dep_temp,
                       linewidth=0, antialiased=False)
# ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlim3d(-max(spray_gun_model.a, spray_gun_model.b), max(spray_gun_model.a, spray_gun_model.b))
ax.set_ylim3d(-max(spray_gun_model.a, spray_gun_model.b), max(spray_gun_model.a, spray_gun_model.b))
plt.show()
