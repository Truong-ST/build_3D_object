import cv2 as cv
import numpy as np
import yaml
import os

cap = cv.VideoCapture('build_3d_object/image/rectangular.mp4')
# cap = cv.VideoCapture(0)
frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
# fps = cap.get(cv.CAP_PROP_FPS)
print(frames)
number_image = 60

# os.mkdir('build_3d_object/image/image_video')
frame_per_image = int(frames/number_image)

file = open("build_3d_object/meshroom/config.yaml", 'r')
parsed_yaml_file = yaml.load(file, Loader=yaml.FullLoader)
print(parsed_yaml_file)
file.close()

file = open('build_3d_object/meshroom/config.yaml', 'w')
parsed_yaml_file['number_image'] = number_image
parsed_yaml_file['frame_per_image'] = frame_per_image
yaml.dump(parsed_yaml_file, file)
file.close()

count = 0
while 1:
    ret, frame = cap.read()
    if ret:
        if count % frame_per_image ==0:
            cv.imwrite('build_3d_object/image/image_video/frame{}.jpg'.format(str(count)), frame)
            # print(count)
        # cv.imshow('frame', frame)
        count += 1
        if cv.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
print('done')