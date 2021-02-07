import trimesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
your_mesh = trimesh.load_mesh('models/wall_type_1.STL')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vertices))

# Auto scale to the mesh size
#scale = your_mesh.vertices.flatten(-1)
#axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()