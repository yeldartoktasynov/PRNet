import numpy as np
from utils.render import render_texture

def get_depth_image(vertices, triangles, h, w, isShow = False):
    z = vertices[:, 2:]
        
    if isShow:
        z = z/max(z)
    depth_image = render_texture(vertices.T, z.T, triangles.T, h, w, 1)
    return np.squeeze(depth_image)