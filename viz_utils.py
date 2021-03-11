from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np


# Drawing arrows: https://stackoverflow.com/a/22867877
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer,)


def plot_path(ax, vertices: [[]], color="g"):

    for i in range(1,len(vertices)):
        a = Arrow3D([vertices[i-1][0], vertices[i][0]],
                    [vertices[i-1][1], vertices[i][1]],
                    [vertices[i-1][2], vertices[i][2]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color=color)
        a.set_zorder(5)
        ax.add_artist(a)


def plot_normals(ax, vertices: [[]], directions: [[]], norm_length = 0.25, color='r') -> None:
    vertices = np.array(vertices)
    directions = np.array(directions)

    for i in range(len(vertices)):
        end = vertices[i] + norm_length*directions[i]
        a = Arrow3D([vertices[i][0], end[0]],
                    [vertices[i][1], end[1]],
                    [vertices[i][2], end[2]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color=color)
        a.set_zorder(5)
        ax.add_artist(a)
