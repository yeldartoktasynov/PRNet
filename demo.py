import numpy as np
import os
from glob import glob
from skimage.io import imread, imsave
from time import time
import argparse
from skimage.transform import rescale
from api import PRN
# from api.PRN import process, get_vertices
from utils.render_app import get_depth_image
import config as cfg
from skimage.transform import estimate_transform, warp
from predictor import PosPrediction


# def process(input, image_info = None):
#     ''' process image with crop operation.
#     Args:
#         input: (h,w,3) array or str(image path). image value range:1~255. 
#         image_info(optional): the bounding box information of faces. if None, will use dlib to detect face. 

#     Returns:
#         pos: the 3D position map. (256, 256, 3).
#     '''
#     if isinstance(input, str):
#         try:
#             image = imread(input)
#         except IOError:
#             print("error opening file: ", input)
#             return None
#     else:
#         image = input

#     if image.ndim < 3:
#         image = np.tile(image[:,:,np.newaxis], [1,1,3])

#     if image_info is not None:
#         bbox = image_info
#         left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
#         old_size = (right - left + bottom - top)/2
#         center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0]) # [161, 118]
#         size = int(old_size*1.6) #528

#     # crop image
#     src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
#     DST_PTS = np.array([[0,0], [0,255], [255, 0]])
#     tform = estimate_transform('similarity', src_pts, DST_PTS)
#     image = image/255.
#     cropped_image = warp(image, tform.inverse, output_shape=(256, 256))
#     # run our net
#     cropped_pos = net_forward(cropped_image)
#     print("FAS4")
#     # restore 
#     cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
#     z = cropped_vertices[2,:].copy()/tform.params[0,0]
#     cropped_vertices[2,:] = 1
#     vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
#     vertices = np.vstack((vertices[:2,:], z))
#     pos = np.reshape(vertices.T, [256, 256, 3])
#     return pos

def process_img(prn, image_path):
    name = image_path.strip().split('/')[-1][:-4]
        # read image
    image = imread(image_path)            
    min_size = min(image.shape[0], image.shape[1])
    if min_size >= 256:
        image = rescale(image, 256./min_size)
        image = (image*255).astype(np.uint8)
    [h, w, c] = image.shape
    if c>3:
        image = image[:,:,:3]
    box = np.array([0, image.shape[0]-1, 0, image.shape[1]-1]) # cropped with bounding box
    pos = prn.process(image, box)
    if pos is None:
        return
    vertices = prn.get_vertices(pos)
    st = time()
    depth_image = get_depth_image(vertices, prn.triangles, h, w)
    end = time()
    print("image name: ", name)
    print("rendering time: ", end-st)
    print("img size: ", (h, w))
    print(depth_image.shape)
    imsave(os.path.join(cfg.save_folder, name + '_depth.jpg'), depth_image)
    # imsave(os.path.join(cfg.save_folder, name + '_depth.jpg'), image)


def main():
    # print(prn.triangles.shape)
    
    prn = PRN()
    # ------------- load data
    image_folder = os.path.join(os.getcwd(), "test/")
    save_folder = os.path.join(os.getcwd(), "test_out/")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))

    for _, im_p in enumerate(image_path_list):
        process_img(prn, im_p)

    # results = Parallel(n_jobs=2)(delayed(process_img)(image_path, save_folder) for _, image_path in enumerate(image_path_list))

if __name__ == '__main__':

    start = time()
    # parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    # parser.add_argument('-i', '--inputDir', default='test/', type=str,
    #                     help='path to the input directory, where input images are stored.')
    # parser.add_argument('-o', '--outputDir', default='test_out', type=str,
    #                     help='path to the output directory, where results(obj,txt files) will be stored.')
    # parser.add_argument('--gpu', default='0', type=str,
    #                     help='set gpu id, -1 for CPU')

    main()

    end = time()

    print("SPENT TIME: ", end - start)
