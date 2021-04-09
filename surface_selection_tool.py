import trimesh
import numpy as np


def get_mask_triangle_indices(mesh):
    faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[1] < -0.1 and normal[0] > -0.5])
    return faces_mask
