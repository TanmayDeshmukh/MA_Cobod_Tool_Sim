from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.text import Annotation


def plot_path(ax, vertices: [[]], color="g", lw=2, hw=0.4):
    for i_ in range(1, len(vertices)):
        a = Arrow3D([vertices[i_ - 1][0], vertices[i_][0]],
                    [vertices[i_ - 1][1], vertices[i_][1]],
                    [vertices[i_ - 1][2], vertices[i_][2]], mutation_scale=10,
                    lw=lw, arrowstyle="-|>, head_width="+str(hw), color=color)
        a.set_zorder(1000)
        ax.add_artist(a)


def plot_normals(ax, vertices: [[]], directions: [[]], norm_length=0.25, color='r', lw=2, hw=0.4) -> None:
    vertices = np.array(vertices)
    directions = np.array(directions)

    for i_ in range(len(vertices)):
        end = vertices[i_] + norm_length * directions[i_]
        a = Arrow3D([vertices[i_][0], end[0]],
                    [vertices[i_][1], end[1]],
                    [vertices[i_][2], end[2]], mutation_scale=10,
                    lw=lw, arrowstyle="-|>, head_width="+str(hw), color=color)
        a.set_zorder(1000)
        ax.add_artist(a)


def visualize_deposition(template, X_grid, Y_grid):
    fig = plt.figure(figsize=(8, 3))
    fig.tight_layout()
    fig.canvas.set_window_title('Surface deposition intensity')
    fig.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=00.0)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X_grid, Y_grid, template,
                     antialiased=False, cmap="coolwarm", lw=0.5, rstride=1, cstride=1, alpha=0.5)
    ax2.contour(X_grid, Y_grid, template, 10, lw=3, colors="k", linestyles="solid")
    # ax.contour(X_grid, Y_grid, template, zdir='z', offset=self.f_max*1.5, cmap="coolwarm")
    # ax2.contour(X_grid, Y_grid, template, zdir='x', offset=-np.min(X_grid[0]), cmap="coolwarm")
    # ax2.contour(X_grid, Y_grid, template, zdir='y', offset=np.max(Y_grid[:, 0]), cmap="coolwarm")
    template = np.fliplr(template)
    ax1.imshow(template, extent=[np.min(X_grid[0]), np.max(X_grid[0]), np.min(Y_grid[:, 0]), np.max(Y_grid[:, 0])])

    limits = np.array([getattr(ax2, f'get_{axis}lim')() for axis in 'xyz'])
    pnp = np.ptp(limits, axis=1)
    pnp[2] = pnp[2]*500
    ax2.set_box_aspect(pnp)

    min_x, max_x = np.min(X_grid[0]), np.max(X_grid[0])
    min_y, max_y = np.min(Y_grid[0]), np.max(Y_grid[0])
    min_z, max_z = np.min(np.min(template, axis = 0)), np.max(np.max(template, axis = 0))

    #ax2.set_xlim(min_x, max_x)
    #ax2.set_ylim(min_z, max_y)
    # ax2.set_zlim(min_z, max_z)

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

def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=20, n_circles_latitude=None):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z


# Drawing arrows: https://stackoverflow.com/a/22867877
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer, )


class Visualizer:
    def __init__(self):
        self.fig_mesh = plt.figure()
        self.axs_mesh = self.fig_mesh.add_subplot(111, projection='3d')
        self.fig_mesh.canvas.set_window_title('Surface mesh')
        self.fig_mesh.tight_layout()
        self.fig_mesh.subplots_adjust(left=-0.15, right=1.05, top=1.1, bottom=0)

        self.fig_slice = plt.figure()
        self.axs_slice = self.fig_slice.add_subplot(111, projection='3d')
        self.fig_slice.canvas.set_window_title('Slicing')
        self.fig_slice.tight_layout()
        self.fig_slice.subplots_adjust(left=-0.15, right=1.05, top=1.1, bottom=0)

        self.fig_unord = plt.figure()
        self.axs_unord = self.fig_unord.add_subplot(111, projection='3d')
        self.fig_unord.canvas.set_window_title('Unordered path')
        self.fig_unord.tight_layout()
        self.fig_unord.subplots_adjust(left=-0.15, right=1.05, top=1.1, bottom=0)

        self.fig_temp = plt.figure()
        self.axs_temp = self.fig_temp.add_subplot(111, projection='3d')
        self.fig_temp.canvas.set_window_title('Ordered path')
        self.fig_temp.tight_layout()
        self.fig_temp.subplots_adjust(left=-0.15, right=1.05, top=1.1, bottom=0)

        self.fig_init = plt.figure()
        self.axs_init = self.fig_init.add_subplot(111, projection='3d')
        self.fig_init.canvas.set_window_title('Initial path')
        self.fig_init.tight_layout()
        self.fig_init.subplots_adjust(left=-0.15, right=1.05, top=1.1, bottom=0)

        self.final_path_fig, self.final_path_ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.final_path_fig.tight_layout()
        self.final_path_fig.subplots_adjust(left=-0.15, right=1.05, top=1.1, bottom=0)
        self.final_path_fig.canvas.set_window_title('Constrained Path')

        self.fig_distrib_hist = plt.figure()
        self.fig_distrib_hist.canvas.set_window_title('Surface distribution')
        self.ax_distrib_hist = self.fig_distrib_hist.add_subplot(111)
        self.ax_distrib_hist.set_xlabel('deposition thickness (mm)')

        self.final_rendering_fig, self.final_rendering_ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.final_rendering_fig.tight_layout()
        self.final_rendering_fig.subplots_adjust(left=-0.1, right=1.1, top=1.1, bottom=-0.05)

        self.all_axs =  [self.axs_init, self.final_path_ax, self.axs_temp, self.axs_unord, self.axs_slice, self.axs_mesh]

    def mesh_view_adjust(self, mesh):
        for ax in self.all_axs:
            # for ax in axr:
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
            min_lim = min(mesh.bounds[0, :])
            max_lim = max(mesh.bounds[1, :])
            ax.set_xlim3d(mesh.bounds[0][0] - 0.5, mesh.bounds[1][0] + 0.5)
            ax.set_ylim3d(mesh.bounds[0][1] - 0.5, mesh.bounds[1][1])
            ax.set_zlim3d(mesh.bounds[0][2], mesh.bounds[1][2] + 0.5)

            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
            ax.set_box_aspect(np.ptp(limits, axis=1))
        # plot the ground plane
        xx, yy = np.meshgrid(np.arange(mesh.bounds[0][0], mesh.bounds[1][0], 0.2),
                             np.arange(mesh.bounds[0][1], mesh.bounds[1][1], 0.2))
        z = np.full((len(xx), len(xx[0])), 0)
        self.final_path_ax.plot_surface(xx, yy, z, alpha=0.5, zorder=-1)
        plt.draw()
        plt.pause(0.001)

    def draw_mesh(self, mesh):
        for ax, col in zip(self.all_axs,
                      ['grey']+[ 'cornflowerblue']*(len(self.all_axs)-1)):

            mplot = mplot3d.art3d.Poly3DCollection(mesh.triangles)
            # mplot.set_alpha(0.8)
            mplot.set_facecolor(col)

            mplot.set_sort_zpos(-1)
            if ax != self.axs_mesh:
                ax.add_collection3d(mplot)
            if ax==self.axs_mesh:
                mplot.set_edgecolor('grey')
                vertices = mesh.vertices
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', s=10)

        plt.draw()
        plt.pause(0.001)

visualizer = None


def open_figures():
    global visualizer
    if not visualizer:
        visualizer = Visualizer()