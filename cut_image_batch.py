import os
import sys

import cv2

folder_fn = sys.argv[1].split('/')[0]
width_start = int(sys.argv[2])
width = int(sys.argv[3])
height_start = int(sys.argv[4])
height = int(sys.argv[5])
output_folder_fn = "{}_cut_{}x{}".format(folder_fn, width, height)
os.system('rm -rf {}'.format(output_folder_fn))
os.system('mkdir {}'.format(output_folder_fn))

all_img = os.listdir(folder_fn)
for img_n in all_img:
    try:
        fn = os.path.join(folder_fn, img_n)
        img = cv2.imread(fn)
        new_img = img[height_start:height_start +
                      height, width_start:width_start+width, :]
        new_img_fn = os.path.join(output_folder_fn, img_n)
        cv2.imwrite(new_img_fn, new_img)
        print("Write new image: {}, img size: {}x{}".format(
            new_img_fn, width, height))
    except:
        print("error, skip")
