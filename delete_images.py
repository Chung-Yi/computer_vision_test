import os
import glob





def delete_image(path):
    for image in glob.glob(path):
        image_name = image.split('/')[2]
        # print(image_name)
        if 'face' in image_name:
            # print(image)
            continue
        else:
            # print('aaaaaa')
            os.remove(image)
        
