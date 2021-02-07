from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from numpy import linalg as LA
import stl
import inspect
import trimesh
import itertools
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import copy
from mpl_toolkits import mplot3d
from matplotlib import pyplot

from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as v
mesh = trimesh.load_mesh('models/wall_type_1.STL')

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
min_lim = min(mesh.bounds[0,:])
max_lim = max(mesh.bounds[1,:])
ax.set_xlim3d(min_lim, max_lim)
ax.set_ylim3d(min_lim,max_lim)
ax.set_zlim3d(min_lim,max_lim)

original_mesh = copy.copy(mesh)
scene = trimesh.Scene()
#scene.add_geometry(original_mesh)
faces_mask = np.array([i for i, normal in enumerate(mesh.face_normals) if normal[2]>0.5])
print('faces_mask', faces_mask)
mesh.update_faces(faces_mask)
mesh.visual.face_colors = [50, 150, 50, 255]
scene.add_geometry(mesh)

y_extents = mesh.bounds[:,0]
# slice every .10 model units
y_levels  = np.arange(*y_extents, step=0.1)#-mesh.bounds[0][0]
# https://github.com/mikedh/trimesh/issues/743#issuecomment-642157661
#mesh.show()
# find a bunch of parallel cross sections
print('mesh.bounds', mesh.bounds, 'y_extents', y_extents, 'y_levels', y_levels)
sections = mesh.section_multiplane(plane_origin=[0,0,0],
                                   plane_normal=[1,0,0],
                                   heights=y_levels)
#sections = [s for s in sections if s]
#print('sections', sections)
sectionsverts = [s.vertices for s in sections]
#print('sectionsverts', sectionsverts)
d3sections = [section.to_3D() for section in sections ]

face_indices = [path.metadata['face_index'] for path in sections]
d3sectionsverts = [s.vertices for s in d3sections]
#print('face_indices', face_indices)
face_normals = [mesh.face_normals[segment_face_indices] for segment_face_indices in face_indices]
#print('face_normals', face_normals)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_path(ax, vertices: [[]]):


    for i in range(1,len(vertices)):
        a = Arrow3D([vertices[i-1][0], vertices[i][0]],
                    [vertices[i-1][1], vertices[i][1]],
                    [vertices[i-1][2], vertices[i][2]], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    vertices = np.transpose(vertices)
    #ax.plot(vertices[0], vertices[1], vertices[2])

standoff_dist = 0.2
vert_dist_threshold = 0.1
vert_iter = 0
direction_flag = False
for section_iter, section_path_group in enumerate(d3sections):
    #path_group.show()
    #print('path_group',type(section_path_group) ,section_path_group, 'path_group attribs:\n', section_path_group.__dict__ , '\n')
    face_indices = section_path_group.metadata['face_index']
    new_entities = []
    all_verts = []
    all_normals = []
    all_text_entities = []
    direction_flag = not direction_flag
    for subpath_iter, subpath in enumerate(section_path_group.entities):
        translated_verts = []
        ori_verts = []
        normals = []
        #print('subpath.points', subpath.points, 'section_path_group attribs:\n', type(section_path_group), section_path_group.__dict__)
        for line_segment_index in range(len(subpath.points)-1):
            this_face_normal = mesh.face_normals[face_indices[line_segment_index]]
            normals.append(this_face_normal)
            normals.append(this_face_normal)
            vert1_index = subpath.points[line_segment_index]
            vert1 = section_path_group.vertices[vert1_index]
            vert2_index = subpath.points[line_segment_index+1]
            vert2 = section_path_group.vertices[vert2_index]
            ori_verts.append(vert1)
            ori_verts.append(vert2)
            new_ver1 = this_face_normal*standoff_dist + vert1
            new_ver2 = this_face_normal * standoff_dist + vert2

            #translated_verts.append([x for x in vert1])
            #translated_verts.append([x for x in vert2])
            translated_verts.append([x for x in new_ver1])
            translated_verts.append([x for x in new_ver2])
            # print('translated_verts', np.array(translated_verts[-1]), str(len(translated_verts)))
        # check first 2 y values
        if (translated_verts[0][1] > translated_verts[1][1]) ^ direction_flag:
            translated_verts.reverse()
            #all_text_entities.append(text_entity)
        #translated_verts = list(set([tuple(x) for x in translated_verts]))

        #print('range(len(translated_verts))', list(range(len(translated_verts))))

        filtered_normals = [normals[0]]
        filtered_trnaslated_verts =[translated_verts[0]]
        filtered_all_text_entities = []
        num = section_iter + len(all_verts) + len(filtered_trnaslated_verts)
        vert_iter += 1
        print('\nvert no.', num, section_iter, len(all_verts) , len(filtered_trnaslated_verts), vert_iter)
        ax.text(filtered_trnaslated_verts[-1][0], filtered_trnaslated_verts[-1][1], filtered_trnaslated_verts[-1][2],
                str(vert_iter), color='red')
        for i in range(1, len(translated_verts)):
            if LA.norm(np.array(translated_verts[i])-np.array(translated_verts[i-1])) > vert_dist_threshold and tuple(normals[i-1])!= tuple(normals[i]) or i==len(translated_verts)-1:
                vert_iter += 1
                filtered_trnaslated_verts.append(translated_verts[i])
                filtered_normals.append(normals[i])
                num = section_iter + len(all_verts) + len(filtered_trnaslated_verts)
                print('vert no.', num, section_iter, len(all_verts) , len(filtered_trnaslated_verts), vert_iter)
                ax.text(filtered_trnaslated_verts[-1][0], filtered_trnaslated_verts[-1][1], filtered_trnaslated_verts[-1][2], str(vert_iter), color='red')

                # text_entity = trimesh.path.entities.Text(origin=0, text='point '+str(len(filtered_trnaslated_verts)), height=0.5, vector=1, normal=0, align=None, layer=None)
                # scene.add_geometry(text_entity)
                #text_entity.show(np.array(filtered_trnaslated_verts))
                #text_entity.points = np.array(list(range(len(filtered_trnaslated_verts))))
                #print('text_entity', type(text_entity),text_entity.__dict__, text_entity.points)
                #filtered_all_text_entities.append(text_entity)
        entity = trimesh.path.entities.Line(points=list(range(len(all_verts), len(filtered_trnaslated_verts) + len(all_verts))))
        new_entities.append(entity)
        all_text_entities += filtered_all_text_entities
        normals = np.array(normals)
        filtered_normals = np.array(filtered_normals)
        all_verts += filtered_trnaslated_verts
        filtered_trnaslated_verts = np.array(filtered_trnaslated_verts)
        translated_verts = np.array(translated_verts)
        vec = np.column_stack((filtered_trnaslated_verts, filtered_trnaslated_verts + (filtered_normals * .08)))
        #print('translated_verts', translated_verts, '\nfiltered_trnaslated_verts\n',filtered_trnaslated_verts,np.transpose(filtered_trnaslated_verts)[0])
        normal_path = trimesh.load_path(vec.reshape((-1, 2, 3))) # returns Path3D object
        normal_path.colors = [trimesh.visual.color.to_rgba([255, 0, 0, 255])]*len(vec)
        #print('normal path', normal_path)
        scene.add_geometry(normal_path)
        plot_path(ax, vertices= filtered_trnaslated_verts)
        #ax.plot(np.transpose(filtered_trnaslated_verts)[0], np.transpose(filtered_trnaslated_verts)[1], np.transpose(filtered_trnaslated_verts)[2])

    all_verts=np.array(all_verts)
    #print('all verts', all_verts, 'new_entities', len(new_entities))
    #print('new_entities+filtered_all_text_entities', new_entities+all_text_entities)
    path3ds = trimesh.path.path.Path3D(entities=new_entities, vertices=np.array(all_verts), metadata={}, colors=[trimesh.visual.color.to_rgba([0, 255, 0, 255])]*len(new_entities))
    scene.add_geometry(path3ds)
    #print('\nori_verts', len(ori_verts), ori_verts, '\ntranslated_verts',len(translated_verts), translated_verts)

#path3ds = trimesh.path.path.Path3D(entities=new_entities, vertices = np., metadata = section_path_group.metadata)

#scene.show()
plt.show()
combined3d = np.sum(d3sections)
#print(sections, sections[-1].vertices)
#print(np.array(sections))
combined = np.sum(sections)
#print(combined, combined3d, combined3d.vertices)
#combined.show()
#combined3d.show()
def displace_segment_along_vector(path3D, vector, distance):
    return vector*distance+path3D

