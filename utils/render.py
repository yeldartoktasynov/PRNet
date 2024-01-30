'''
Author: YadiraF 
Mail: fengyao@sjtu.edu.cn
'''
import numpy as np
from time import time
from joblib import Parallel, delayed

def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y] 
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]
    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)
    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno
    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


def process_triangles(image, vertices, tri_depth, tri_tex, depth_buffer, h, w, tri, i):

    # the inner bounding box
    umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
    umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

    vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
    vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

    if (umax>=umin) and (vmax>=vmin):    
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u,v], vertices[:2, tri]): 
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]


def render_texture(vertices, colors, triangles, h, w, c = 3):
    ''' render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width    
    '''
    # initial 
    image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3. 
    tri_tex = (colors[:, triangles[0,:]] + colors[:,triangles[1,:]] + colors[:, triangles[2,:]])/3.

    # results = Parallel(n_jobs=2)(delayed(process_img)(image_path, prn, save_folder) for _, image_path in enumerate(image_path_list))


    for i in range(triangles.shape[1]):
        tri = triangles[:, i] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u,v], vertices[:2, tri]): 
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image
