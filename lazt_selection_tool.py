import trimesh
import numpy as np

mesh = trimesh.load_mesh('models/wall_type_1.STL')
print('mesh.faces', mesh.faces)
print('mesh.facets', mesh.facets)
print('mesh.ver', mesh.vertices)
print('mesh.face_normals', mesh.face_normals)
print('mesh.facets_normal', mesh.facets_normal)
print('face_adjacency', mesh.face_adjacency[:5])
print('face_adjacency_edges', mesh.face_adjacency_edges[:5])
print('face_adjacency_unshared', mesh.face_adjacency_unshared[:5])
check = np.column_stack((mesh.face_adjacency_edges,
                         mesh.face_adjacency_unshared))
# make sure rows are sorted
check.sort(axis=1)

# find the indexes of unique face adjacency
adj_unique = trimesh.grouping.unique_rows(check)[0]

# find the unique indexes of the original faces
faces_mask = trimesh.grouping.unique_bincount(
    mesh.face_adjacency[adj_unique].reshape(-1))
print('faces_mask', faces_mask, type(faces_mask))
faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[2]>0.5])
print('faces_mask', faces_mask)
# apply the mask to remove non-unique faces
mesh.update_faces(faces_mask)
"""
new_faces = new_verts = new_norms = []
for face, vertices, normal in zip(mesh.faces, mesh.vertices, mesh.face_normals):
    if normal[2]>0:
        new_faces.append(face)
        new_norms.append(normal)
        new_verts.append(vertices)

new_mesh = trimesh.Trimesh(faces=new_faces, vertices=new_verts)#, face_normals=new_norms)
"""
mesh.show()
