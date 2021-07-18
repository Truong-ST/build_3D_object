import numpy as np
import math
import pyvista as pv

pi = math.pi
points = []
for i in np.arange(0, pi, 0.2):
    z = math.cos(i)
    R = math.sin(i)
    for j in np.arange(0, 2*pi, 0.2):
        points.append(np.array([R*math.cos(j), R*math.sin(j), z]))
print(points[:5])

# points is a 3D numpy array (n_points, 3) coordinates of a sphere
cloud = pv.PolyData(points)
cloud.plot()

volume = cloud.delaunay_3d(alpha=2.)
shell = volume.extract_geometry()
shell.plot()

# faces = np.array([(4,0,3,4,1), (4,1,4,5,2)])
# mesh = pv.PolyData(point_cloud, faces)

# # Create a plotting window and display!
# p = pv.Plotter()

# # Add the mesh and some labels
# p.add_mesh(mesh, show_edges=True)
# # p.add_point_labels(mesh.points, ["%d"%i for i in range(mesh.n_points)])

# # A pretty view position
# p.camera_position = [(-11.352247399703748, -3.421477319390501, 9.827830270231935),
#  (-5.1831825, -1.5, 1.9064675),
#  (-0.48313206526616853, 0.8593146723923926, -0.16781448484204659)]

# # Render it!
# p.show()