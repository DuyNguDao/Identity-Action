import cv2
from yolov5_face.detect_face import Y5DetectFace, draw_result
from face_recognition.face import Face_Model
from yolov7_pose.detect_pose import Y7Detect, draw_kpts, draw_boxes
import time
import numpy as np
import math
from numpy import random
from strong_sort.strong_sort import StrongSORT
from pathlib import Path
from collections import deque
import torch
import argparse
from classification_lstm.utils.load_model import Model
import random
import sys
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGTHS = ROOT

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(torch.cuda.is_available())


def compute_distance(nose, list_face):
    list_face = list_face.astype('float')
    # get center top
    top_center = list_face[:, :2] + np.concatenate([list_face[:, 2:3], list_face[:, 1:2]], axis=1)
    top_center /= 2
    distance = np.sqrt(np.sum((top_center - nose)**2, axis=1))
    d_min = np.amin(distance)
    idx = np.argmin(distance)
    return d_min, idx


def detect_video(url_video=None, flag_save=False, fps=None, name_video='video.avi'):

    # ******************************** LOAD MODEL *************************************************
    # load model detect yolov7 pose
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y7_pose = Y7Detect(weights=WEIGTHS / 'yolov7_pose/weights/yolov7_w6_pose.pt')
    class_name = y7_pose.class_names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_name]

    # *************************** LOAD MODEL LSTM ************************************************
    action_model = Model(WEIGTHS / 'classification_lstm/weights/best_skip.pt')

    face_model = Face_Model()

    # **************************** INIT TRACKING *************************************************
    tracker = StrongSORT(device=device, max_age=30, n_init=3, max_iou_distance=0.7)  # deep sort

    # ********************************** READ VIDEO **********************************************
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    h_norm, w_norm = 720, 1280
    if frame_height > h_norm and frame_width > w_norm:
        frame_width = w_norm
        frame_height = h_norm
    # get fps of camera
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    # save video
    if flag_save is True:
        video_writer = cv2.VideoWriter(name_video,
                                       cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # ******************************** REAL TIME ********************************************
    memory = {}
    count = True
    turn_detect_face = True
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        # count += 1
        if h > h_norm and w > w_norm:
            frame = cv2.resize(frame, (w_norm, h_norm), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        frame[0:h-550, w-300:w] = np.zeros((h-550, 300, 3), dtype='uint8')

        # **************************** detect pose ***************************************
        if count:
            bbox, label, score, label_id, kpts, scores_pt = y7_pose.predict(frame)
            bbox, score, kpts = np.array(bbox), np.array(score), np.array(kpts)

        # detect face and recognition
        bbox_f, label_f, label_id_f, score_f, landmark_f = face_model.detect(frame)
        face = {}
        if turn_detect_face:
            for idx, box in enumerate(bbox_f):
                feet = face_model.face_encoding(frame, kps=np.array(landmark_f[idx]))
                name = face_model.face_compare(feet, threshold=0.3)
                face.update({name: box})
            turn_detect_face = False
            # draw_result(frame, box, name, score_f[idx], landmark_f[idx])

        if len(bbox) != 0:
            if count:
                data = tracker.update(bbox, score, kpts, frame)
            for outputs in data:
                if len(outputs['bbox']) != 0:
                    box, kpt, track_id, list_kpt = outputs['bbox'], outputs['kpt'], outputs['id'],\
                                                             outputs['list_kpt']
                    list_face = list(face.values())
                    if str(track_id) not in memory:
                        if len(list_face) == 0:
                            memory.update({str(track_id): 'Unknown'})
                            turn_detect_face = True
                        else:
                            d_min, pos = compute_distance(np.array(kpt[0]), np.array(list_face))
                            w_min = (list_face[pos][3]-list_face[pos][1]) + (list_face[pos][2]-list_face[pos][0])
                            if d_min > w_min:
                                memory.update({str(track_id): 'Unknown'})
                                turn_detect_face = True
                            else:
                                memory.update({str(track_id): list(face.keys())[pos]})
                    else:
                        if memory[str(track_id)] == 'Unknown':
                            turn_detect_face = True
                            if len(list_face) != 0:
                                d_min, pos = compute_distance(np.array(kpt[0]), np.array(list_face))
                                w_min = (list_face[pos][3]-list_face[pos][1]) + (list_face[pos][2]-list_face[pos][0])
                                if d_min <= w_min:
                                    memory.update({str(track_id): list(face.keys())[pos]})
                                    turn_detect_face = False
                                else:
                                    turn_detect_face = True
                    name = memory[str(track_id)]
                    icolor = class_name.index('0')
                    # draw_boxes(frame, box, color=colors[icolor])
                    draw_kpts(frame, [kpt])
                    color = (0, 255, 0)
                    if len(list_kpt) == 15:
                        action, score = action_model.predict([list_kpt], w, h, batch_size=1)
                    try:
                        if action[0] == "Fall Down":
                            color = (0, 0, 255)
                        cv2.putText(frame, '{}: {} - {}'.format(name, action[0], track_id),
                                    (max(box[0]-20, 0), box[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                        action = ["Pending..."]
                    except:
                        cv2.putText(frame, '{}: {} - {}'.format(name, "Pending...", track_id),
                                    (max(box[0]-20, 0), box[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        count = not count
        # ******************************************** SHOW *******************************************
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        if flag_save is True:
            video_writer.write(frame)

    cap.release()
    if flag_save is True:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-op", "--option", help="if save video then choice option = 1", default=True, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='recog_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=30, help="FPS of output video", type=int)
    args = parser.parse_args()

    # PATH VIDEO
    url = '/home/duyngu/Downloads/video_test/20221001153808324_7F01683RAZE9C1D.mp4'
    source = args.file_name
    cv2.namedWindow('video')
    # if run  as terminal, replace url = source
    detect_video(url_video=url, flag_save=args.option, fps=args.fps, name_video=args.output)
