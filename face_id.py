import face_recognition as fr
import cv2
import pickle
import os
import sys
import glob
import logging
import delete_images as dl


def whosface(image, known_face_encodings, face_location, tolerance):
    face_encodings = fr.face_encodings(image, face_location)[0]
    face_distances = fr.face_distance(known_face_encodings, face_encodings)
    # print(face_distances)
    min_distance_index = face_distances.argmin()
    min_distance = face_distances[min_distance_index]
    # min_distance_index = face_distances.index(min_distance)
    # print(min_distance)
    
    if min_distance < tolerance:
        return min_distance_index
    else:
        return False

    # for i, face_distance in enumerate(face_distances):
    #     count+=1
    #     if face_distance < 0.38:
    #         face_dis.append(face_distance)
    #         print(face_distance)
    #     else:
    #         pass
    #     print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    #     print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    # print(count)
    # print(face_dis)

    # matches = fr.compare_faces(known_face_encodings, face_encodings, tolerance)
    # print(matches)
    # return matches

def main():
    f_enc = open(os.path.join('astra_face_db','encode_list.pkl'), 'rb')
    f_name = open(os.path.join('astra_face_db','name_list.pkl'), 'rb')
    # print(f_enc)
    known_face_encodings = pickle.load(f_enc)
    known_face_names = pickle.load(f_name)
    # print(known_face_names)
    # print('AAAAA',len(known_face_encodings))
    f_enc.close()
    f_name.close()

    logging.info('Model loading finish!')

    folder = sys.argv[1]
    tolerance = float(sys.argv[2])
    path = 'face_images/tmp{}/*.jpg'.format(folder)
    

    face_names = []
    input_name = []
    
    
    dl.delete_image(path)
    correct_identity = 0
    for image in glob.glob(path):
        # print(image)
        image_name = image.split('/')[2].split('_')[0]
        if len(image_name) > 0 and len(image_name) < 3:
            image_name.upper()
        else:
            image_name = image_name.capitalize()
        input_name.append(image_name)
        #print(image_name)
        image = cv2.imread(image)
        image_rgb = image[:, :, ::-1]
        image_size = image.shape
        h, w = image_size[0], image_size[1]

        face_index = whosface(image_rgb, known_face_encodings, [(0,w,h,0)], tolerance)
        

        name = 'Unknown'
        
        if face_index:
            # first_match_index = face.index(True)
            name = known_face_names[face_index]
            # print(image_name, name)
            # pass

            if image_name in name:
                # print(image_name, name)
                correct_identity += 1
            else:
                pass
        face_names.append(name)
    
    accuracy = correct_identity/len(input_name)
    
    print('distance:', folder)
    print('tolerance:', tolerance)
    print('correct_identity', correct_identity)
    print('accuracy:{0:.2f}'.format(accuracy))
    print(input_name)
    print(face_names)
    






if __name__ == '__main__':
    main()