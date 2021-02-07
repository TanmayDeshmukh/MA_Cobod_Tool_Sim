import stl
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
your_mesh = stl.mesh.Mesh.from_file('models/wall_type_1.STL')
print('your_mesh.vectors', your_mesh.vectors)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
#scale = your_mesh.points.flatten(-1)
#axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()