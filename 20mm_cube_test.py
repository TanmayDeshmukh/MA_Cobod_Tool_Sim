from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import inspect
import trimesh

from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as v
mesh = trimesh.load_mesh('models/20mm_cube.stl')
#faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[2]>0.5])
#print('faces_mask', faces_mask)
#mesh.update_faces(faces_mask)

y_extents = mesh.bounds[:,2]
# slice every .10 model units
y_levels  = np.arange(*y_extents, step=10)#-mesh.bounds[0][0]
# https://github.com/mikedh/trimesh/issues/743#issuecomment-642157661
mesh.show()
# find a bunch of parallel cross sections
print('mesh.bounds', mesh.bounds, 'y_extents', y_extents, 'y_levels', y_levels)
sections = mesh.section_multiplane(plane_origin=[0,0,0],
                                   plane_normal=[0,0,1],
                                   heights=y_levels)
#sections = [s for s in sections if s]
print('sections', sections)
sectionsverts = [s.vertices for s in sections]
print('sectionsverts', sectionsverts)
d3sections = [section.to_3D() for section in sections ]

face_indices = [path.metadata['face_index'] for path in sections]
d3sectionsverts = [s.vertices for s in d3sections]
print('face_indices', face_indices)
combined3d = np.sum(d3sections)
#print(sections, sections[-1].vertices)
#print(np.array(sections))
combined = np.sum(sections)
print(combined, combined3d, combined3d.vertices)
#combined.show()
combined3d.show()


