from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import inspect
import trimesh
import itertools

from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as v
mesh = trimesh.load_mesh('models/wall_type_1.STL')
faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[2]>0.5])
print('faces_mask', faces_mask)
mesh.update_faces(faces_mask)

y_extents = mesh.bounds[:,0]
# slice every .10 model units
y_levels  = np.arange(*y_extents, step=0.5)#-mesh.bounds[0][0]
# https://github.com/mikedh/trimesh/issues/743#issuecomment-642157661
mesh.show()
# find a bunch of parallel cross sections
print('mesh.bounds', mesh.bounds, 'y_extents', y_extents, 'y_levels', y_levels)
sections = mesh.section_multiplane(plane_origin=[0,0,0],
                                   plane_normal=[1,0,0],
                                   heights=y_levels)
#sections = [s for s in sections if s]
print('sections', sections)
sectionsverts = [s.vertices for s in sections]
print('sectionsverts', sectionsverts)
d3sections = [section.to_3D() for section in sections ]

face_indices = [path.metadata['face_index'] for path in sections]
d3sectionsverts = [s.vertices for s in d3sections]
print('face_indices', face_indices)
face_normals = [mesh.face_normals[segment_face_indices] for segment_face_indices in face_indices]
print('face_normals', face_normals)


standoff_dist = 1
for section_iter, section_path_group in enumerate(d3sections):
    #path_group.show()
    #print('path_group',type(section_path_group) ,section_path_group, 'path_group attribs:\n', section_path_group.__dict__ , '\n')
    face_indices = section_path_group.metadata['face_index']

    for subpath_iter, subpath in enumerate(section_path_group.entities):
        translated_verts = []
        ori_verts = []
        print('subpath.points', subpath.points)
        for line_segment_index in range(len(subpath.points)-1):
            this_face_normal = mesh.face_normals[face_indices[line_segment_index]]
            vert1_index = subpath.points[line_segment_index]
            vert1 = section_path_group.vertices[vert1_index]
            vert2_index = subpath.points[line_segment_index+1]
            vert2 = section_path_group.vertices[vert2_index]
            ori_verts.append(vert1)
            ori_verts.append(vert2)
            new_ver1 = this_face_normal*standoff_dist + vert1
            new_ver2 = this_face_normal * standoff_dist + vert2

            translated_verts.append(new_ver1)
            translated_verts.append(new_ver2)

        print('\nori_verts', len(ori_verts), ori_verts, '\ntranslated_verts',len(translated_verts), translated_verts)

combined3d = np.sum(d3sections)
#print(sections, sections[-1].vertices)
#print(np.array(sections))
combined = np.sum(sections)
#print(combined, combined3d, combined3d.vertices)
#combined.show()
combined3d.show()
def displace_segment_along_vector(path3D, vector, distance):
    return vector*distance+path3D

