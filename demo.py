import numpy as np
import os
from glob import glob
from skimage.io import imread, imsave
from time import time
import argparse
from skimage.transform import resize
from api import PRN
from utils.render_app import get_depth_image

import threading


def main(args):

    start = time()

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN()

    # ------------- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))

    for _, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]
        # read image
        image = imread(image_path)
        # image = resize(image, (image.shape[0] // 4, image.shape[1] // 4),
        #                anti_aliasing=True)
        [h, w, c] = image.shape

        if c>3:
            image = image[:,:,:3]

        # the core: regress position map
        # if image.shape[0] == image.shape[1]:
        # image = resize(image, (256,256))
        # pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
        # else:
        box = np.array([0, image.shape[0]-1, 0, image.shape[1]-1]) # cropped with bounding box
        pos = prn.process(image, box)
        
        if pos is None:
            continue
        
        vertices = prn.get_vertices(pos)
        st = time()
        depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
        end = time()
        print("rendering time: ", end-st)
        print("img size: ", (h, w))
        imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)


    end = time()
    print("SPENT TIME: ", end - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='test/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='test_out', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')

    main(parser.parse_args())
