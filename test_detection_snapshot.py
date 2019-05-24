from matplotlib import pyplot as plt
import cv2
import person_recognition as pr
import sys
import timeit
import face_recognition as fr
import numpy as np
import os
import uuid
import logging


def draw_ui(frame, persons):
    frame_ = np.copy(frame)/255
    for person in persons:
        x1, y1 = person[0][0], person[0][1]
        x2, y2 = person[1][0], person[1][1]
        cv2.rectangle(frame_, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(frame_)
    plt.show()


def save_ui(frame, persons):
    frame_ = np.copy(frame)
    for person in persons:
        x1, y1 = person[0][0], person[0][1]
        x2, y2 = person[1][0], person[1][1]
        cv2.rectangle(frame_, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('tmp/{}.jpg'.format(uuid.uuid4()), frame_[:, :, ::-1])


def test(frame, ssd_thr, filename, mode=None):
    t1 = timeit.default_timer()
    person_origin = pr.person_detect(frame, mode=mode, ssd_thr=ssd_thr)
    t2 = timeit.default_timer() - t1
    print('Original person detection time elaspsed: {} sec'.format(round(t2, 3)))
    # draw_ui(frame, person_origin)
    save_ui(frame, person_origin)

    # Check how many people and print the human size
    for index, ii in enumerate(person_origin):
        x1, y1, x2, y2 = ii[0][0], ii[0][1], ii[1][0], ii[1][1]
        cv2.imwrite("tmp/{}_{}_person.jpg".format(filename, index),
                    frame[y1:y2, x1:x2, ::-1])
        print("Person size: {}x{}".format(x2 - x1, y2 - y1))

    t1 = timeit.default_timer()
    person_origin_data = pr.person_vector(frame, mode=mode, ssd_thr=ssd_thr)
    people_count = 0
    face_count = 0
    for p_index, ii in enumerate(person_origin_data):
        people_count += 1
        face_locations = fr.face_locations(ii[1])
        for f_index, loc in enumerate(face_locations):
            fx1, fy1, fx2, fy2 = loc[3], loc[0], loc[1], loc[2]
            print("Face area: {}x{}".format(fx2 - fx1, fy2 - fy1))
            cv2.imwrite("tmp/{}_{}_{}_face.jpg".format(filename,
                                                       p_index, f_index),  ii[1][fy1:fy2, fx1:fx2, ::-1])
            enc = fr.face_encodings(ii[1], face_locations)
            face_count += 1
    t2 = timeit.default_timer() - t1
    print('Original person vector, time: {},  people count: {}, face count: {}'.format(
        round(t2, 3), people_count, face_count))

    return (t2, people_count, face_count)


os.system('rm -rf tmp')
os.system('mkdir tmp')

origin_test_overall_time = 0
file_dir = sys.argv[1]
img_name_list = os.listdir(file_dir)
overall_count = 0
original_overall_people_count = 0
original_overall_face_count = 0
ssd_thr = float(sys.argv[2])

test_mode = sys.argv[3]


print("\n\n################################# Test start #################################\n\n")
for img in img_name_list:
    fn = os.path.join(file_dir, img)
    print(fn)
    frame = cv2.imread(fn)[:, :, ::-1]
    overall_count += 1
    print("#################### img: {}, ssd: {} ######################".format(
        img, ssd_thr))
    origin_time, people_count, face_count = test(
        frame, ssd_thr, img, mode=test_mode)
    origin_test_overall_time += origin_time
    original_overall_people_count += people_count
    original_overall_face_count += face_count

    print("#################### Test finished ######################\n\n")

print("Distance folder: {}, ssd threshold: {}, frame size: {}x{}".format(
    file_dir, ssd_thr, frame.shape[1], frame.shape[0]))
print("Overall time analysis: original time: {} sec".format(
    round(origin_test_overall_time / overall_count, 5), 0))
print("original people OK ratio: {}/{}, face OK ratio: {}/{}".format(
    original_overall_people_count, overall_count, original_overall_face_count, overall_count))
