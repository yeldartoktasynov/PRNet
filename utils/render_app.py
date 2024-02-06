import numpy as np
from utils.render import render_texture
from face3d import mesh

def get_depth_image(vertices, triangles, h, w, isShow = False):
    # z = vertices[:, 2:]
        
    # if isShow:
    #     z = z/max(z)
    # depth_image = render_texture(vertices.T, z.T, triangles.T, h, w, 1)
    z = vertices[:,2:]
    z = z - np.min(z)
    z = z/np.max(z)
    attribute = z
    depth_image = mesh.render.render_colors(vertices, triangles, attribute, h, w, c=1)
    return np.squeeze(depth_image)