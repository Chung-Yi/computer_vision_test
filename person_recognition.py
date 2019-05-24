import cv2
import numpy as np
import tensorflow as tf
# import models.reid_model.heads.fc1024 as head
# import models.reid_model.nets.mobilenet_v1_1_224 as model
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

path = os.path.split(__file__)[0]
# Tensorflow human re-ID feature descriptor model
filename = 'models/person_vector.pb'
graph_def = tf.GraphDef()
with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
images = sess.graph.get_tensor_by_name('Placeholder:0')
embeddings = sess.graph.get_tensor_by_name('head/emb/BiasAdd:0')

# tf.Graph().as_default()
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# images = tf.zeros([1, 256, 128, 3], dtype=tf.float32)
# endpoints, body_prefix = model.endpoints(images, is_training=False)
# with tf.name_scope('head'):
#     endpoints = head.head(endpoints, 128, is_training=False)
# tf.train.Saver().restore(sess, os.path.join(
#     path, 'models/reid_model/model/checkpoint-25000'))

# caffe mobilenet models
net = cv2.dnn.readNetFromCaffe(
    os.path.join(path, 'models/mobilenet_model/deploy.prototxt'),
    os.path.join(path, 'models/mobilenet_model/MobileNetSSD_deploy.caffemodel'))
classNames = {15: 'person'}


def person_vector(frame, mode=None, person_locations=None, ssd_thr=0.9):
    person_locations = person_detect(frame, ssd_thr, mode)
    person_images = np.zeros((len(person_locations), 256, 128, 3))
    output_list = list()
    for i, loc in enumerate(person_locations):
        p_img = crop_image(frame, loc)
        resize_img = cv2.resize(p_img, (128, 256))
        person_images[i, :, :, :] = resize_img
        item = [loc, p_img]
        output_list.append(item)
    embs = sess.run(embeddings, feed_dict={images: person_images})

    for i, emb in enumerate(embs):
        output_list[i].append(emb)

    return output_list


# def person_distance(enc1, enc2):
#     return np.sqrt(np.sum(np.square(enc1 - enc2)))

def crop_image(frame, location):
    #   +-----------------------------> X
    #   |
    #   |    left_bottom(x1,y1)
    #   |           *-----------
    #   |           |          |
    #   |           | (person) |
    #   |           |          |
    #   |           |          |
    #   |           |          |
    #   |           |          |
    #   |           |          |
    #   |           |          |
    #   |           |          |
    #   |           |          |
    #   |           |          |
    #   |           |----------*
    #   |                       right_top(x2,y2)
    #   |
    #   ˇ
    #   Y

    left_bottom = location[0]
    right_top = location[1]

    # frame.size=(720, 1280, 3), [y1:y2, x1:x2, 3 channels]
    sub_frame = frame[left_bottom[1]:right_top[1],
                      left_bottom[0]:right_top[0], :]
    return sub_frame


def person_distance(human_encs, human_enc_to_compare):
    if len(human_encs) == 0:
        return np.empty((0))

    return np.linalg.norm(human_encs - human_enc_to_compare, axis=1)


def _person_detect_unit(frame, ssd_thr):
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(
        frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), True)
    net.setInput(blob)
    # Prediction of network
    detections = net.forward()
    # Size of frame resize (300x300)
    rows = frame_resized.shape[0]
    cols = frame_resized.shape[1]
    output = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence of prediction
        if confidence > ssd_thr:  # Filter prediction
            class_id = int(detections[0, 0, i, 1])  # Class label

            # Object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # Factor for scale to original size of frame
            heightFactor = frame.shape[0] / 300.0
            widthFactor = frame.shape[1] / 300.0
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widthFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)
            if class_id in classNames:
                if (xLeftBottom <= 0) or (xRightTop <= 0) or (yLeftBottom <= 0) or (yRightTop <= 0):
                    print('Invalid person image size, skip')
                    continue
                output.append(
                    [(xLeftBottom, yLeftBottom), (xRightTop, yRightTop)])
    return output


def _cal_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
    boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def _axis_transform(origin_coord_loc, transform_coord):
    tran_x, tran_y = transform_coord[0], transform_coord[1]
    x1, y1, x2, y2 = origin_coord_loc[0][0], origin_coord_loc[0][1], origin_coord_loc[1][0], origin_coord_loc[1][1]
    return [(x1 + tran_x, y1 + tran_y), (x2 + tran_x, y2 + tran_y)]


def _cal_area_from_coord(coord):
    return (coord[1][1] - coord[0][1]) * (coord[1][0] - coord[0][0])


def person_detect(frame, ssd_thr=0.5, mode=None):
    if (mode == 'o') or (mode is None):
        return _person_detect_unit(frame, ssd_thr)
    elif mode == 's':
        left_frame = frame[:, :720, :]
        right_frame = frame[:, 560:, :]
        left_output_locations = _person_detect_unit(left_frame, ssd_thr)
        right_output_locations = [_axis_transform(
            right, (561, 0)) for right in _person_detect_unit(right_frame, ssd_thr)]
        output = left_output_locations + right_output_locations
        # replicate_loc_index_list = list()
        # for i, loc_i in enumerate(output):
        #     for j, loc_j in enumerate(output):
        #         iou = _cal_iou(loc_j, loc_i)
        #         if (iou < 1) and (iou > 0.3):
        #             if _cal_area_from_coord(loc_i) > _cal_area_from_coord(loc_j):
        #                 replicate_loc_index_list.append(j)
        #             else:
        #                 replicate_loc_index_list.append(i)

        # for i in reversed(replicate_loc_index_list):
        #     output.pop(i)

        return output
