import cv2 as cv
import numpy as np
import yaml


file = open("build_3d_object/meshroom/config.yaml", 'r')
parsed_yaml_file = yaml.safe_load(file)
file.close()
number_image = parsed_yaml_file['number_image']
frame_per_image = parsed_yaml_file['frame_per_image']


# n = 60
def list_image_matching(number_image, frame_per_image):
    list_image = ['frame{}'.format(i*frame_per_image) for i in range(number_image)]
    pair_image = []
    for i in range(len(list_image)-1):
        pair_image.append((list_image[i], list_image[i+1]))
    return pair_image
