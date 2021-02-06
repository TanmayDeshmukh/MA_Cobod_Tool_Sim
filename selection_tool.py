from pyrender import Mesh, Scene, Viewer
import trimesh

tm = trimesh.load_mesh('./models/fuze.obj')
mesh = Mesh.from_trimesh(tm)
scene = Scene()
scene.add(mesh)

class MyViewer(Viewer):

    def on_mouse_press(self, x, y, buttons, modifiers):
        print('MOUSE PRESSED!', x, y)
        super(MyViewer, self).on_mouse_press(x, y, buttons, modifiers)

v = MyViewer(scene, use_raymond_lighting=True)

scene.camera_rays()