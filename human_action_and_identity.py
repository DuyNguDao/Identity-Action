"""
IDENTITY RECOGNITION AND APPLICATION ACTIONS FOR FALL DETECTION IN THE ELDERLY
Member: DAO DUY NGU, LE VAN THIEN
Mentor: PhD. TRAN THI MINH HANH
Time: 12/11/2022
contact: ddngu0110@gmail.com, ngocthien3920@gmail.com
"""

import cv2
from face_recognition.face import Face_Model
from yolov7_pose.detect_pose import Y7Detect, draw_kpts, draw_boxes
from yolov5_face.detect_face import draw_result
import time
import numpy as np
from numpy import random
from strong_sort.strong_sort import StrongSORT
from track_sort.Sort import SORT
from pathlib import Path
import torch
from classification_lstm.utils.load_model import Model
from classification_stgcn.Actionsrecognition.ActionsEstLoader import TSSTG
import random
from playsound import playsound
from multiprocessing import Process
from database.interface_sql import *
from datetime import datetime, timedelta

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGTHS = ROOT

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(torch.cuda.is_available())


class ActionAndIdentityRecognition:
    def __init__(self):
        # load model detect yolov7 pose
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.y7_pose = Y7Detect()
        self.class_name = self.y7_pose.class_names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_name]
        # *************************** LOAD MODEL LSTM OR ST-GCN ************************************************
        # LSTM
        # self.action_model = Model(device=device, skip=True)
        # ST-GCN
        self.action_model = TSSTG(device=device, skip=True)

        # *************************** LOAD MODEL FACE RECOGNITION ************************************
        self.face_model = Face_Model(device=device)

        # **************************** INIT TRACKING *************************************************
        self.tracker = StrongSORT(device=device, max_age=30, n_init=3, max_iou_distance=0.5)  # deep sort
        # self.tracker = SORT(max_age=30, n_init=1, max_iou_distance=0.7)  # sort

        # ******************************** INIT DATA ********************************************
        self.memory = {}  # memory contain identification human action
        self.memory1 = {}  # memory contain id, face
        self.memory_prob = {}   # memory fall down
        self.prob = 5
        self.turn_detect_face = True  # flag turn on, off face recognition
        self.data = None  # buffer data for skip when tracking
        self.bbox = None  # buffer data for skip when skeleton detection
        self.face_unkhow = cv2.imread('icon/unknown_person.jpg')
        self.face_unkhow = cv2.resize(self.face_unkhow, (112, 112))

    def processing(self, frame, skip=True):
        h, w, _ = frame.shape
        image_save = frame.copy()
        info = {}
        if skip:
            # **************************** SKELETON DETECTION *************************************
            bbox, score, kpts = self.pose_detection(frame)
            self.bbox = bbox
        # ******************************** FACE RECOGNITION ***************************************
        face, frame, id_fc = self.face_recognition(frame)
        # ***************************** TRACKING **************************************************
        if len(self.bbox) != 0:
            if skip:
                self.data = self.tracker.update(self.bbox, score, kpts, frame)
            for outputs in self.data:
                if len(outputs['bbox']) != 0:
                    box, kpt, track_id, list_kpt = outputs['bbox'], outputs['kpt'], outputs['id'], \
                                                   outputs['list_kpt']
                    list_face = np.array(list(face.values()))
                    kpt = kpt[:, :2].astype('int')
                    # ************************************ CHECK ID *******************************************
                    if str(track_id) not in self.memory:
                        self.memory_prob.update({str(track_id): 0})
                        if len(list_face) == 0:
                            self.memory.update({str(track_id): ['Unknown', 0]})
                            self.memory1.update({str(track_id): ['None', self.face_unkhow]})
                            self.turn_detect_face = True
                        else:
                            d_min, pos = self.compute_distance(np.array(kpt[0]), list_face[:, 2, :])
                            w_min = np.sqrt(np.sum((list_face[pos, 1, :] - list_face[pos, 0, :]) ** 2, axis=0))
                            if d_min > w_min:
                                self.memory.update({str(track_id): ['Unknown', 0]})
                                self.memory1.update({str(track_id): ['None', self.face_unkhow]})
                                self.turn_detect_face = True
                            else:
                                self.memory.update({str(track_id): [list(face.keys())[pos], 0]})
                                self.memory1.update({str(track_id): list(id_fc.values())[pos]})
                    else:
                        self.memory.update({str(track_id): [self.memory[str(track_id)][0], 0]})
                        if self.memory[str(track_id)][0] == 'Unknown':
                            self.turn_detect_face = True
                            if len(list_face) != 0:
                                d_min, pos = self.compute_distance(np.array(kpt[0]), list_face[:, 2, :])
                                w_min = np.sqrt(np.sum((list_face[pos, 1, :] - list_face[pos, 0, :]) ** 2, axis=0))
                                if d_min <= w_min:
                                    self.memory.update({str(track_id): [list(face.keys())[pos], 0]})
                                    self.memory1.update({str(track_id): list(id_fc.values())[pos]})
                                    self.turn_detect_face = False
                                else:
                                    self.turn_detect_face = True
                    # get name id
                    name = self.memory[str(track_id)][0]
                    icolor = self.class_name.index('0')
                    # draw_boxes(frame, box, color=colors[icolor])
                    draw_kpts(frame, [kpt])
                    # ************************************ PREDICT ACTION ********************************
                    action = None
                    if len(list_kpt) == 15:
                        # LSTM
                        # action, score = self.action_model.predict([list_kpt], w, h, batch_size=1)

                        # ST-GCN
                        torch.cuda.reset_peak_memory_stats()
                        action, score = self.action_model.predict(list_kpt, (w, h))
                        if action[0] == "Fall Down":
                            self.memory_prob.update({str(track_id): self.memory_prob[str(track_id)] + 1})

                            if self.memory_prob[str(track_id)] == self.prob:
                                now = datetime.now()
                                info.update({str(track_id): {'id': self.memory1[str(track_id)][0],
                                                             'image': self.memory1[str(track_id)][1],
                                                             'name': name, 'action': action[0],
                                                             'time': now.strftime('%a %H:%M:%S')}})
                                # turn on buzzer
                                t = Process(target=playsound, args=('icon/sound_beep-08.mp3',))
                                t.start()
                                image_fd = image_save[max(box[1]-10, 0):min(box[3] + 10, h), box[0]:box[2]]
                                image_fd = cv2.resize(image_fd, (112, 112))
                                add_action(data_tuple=(self.memory1[str(track_id)][0], name, self.memory1[str(track_id)][1]
                                                       , action[0], image_fd, now.strftime('%a %H:%M:%S')),
                                           name_table='action_data')
                        else:
                            self.memory_prob.update({str(track_id): 0})
                    frame = self.draw_frame(frame, box, action, name, track_id)

            # ************************ UPDATE COUNT MEMORY WITH TRACK ID ******************************
            keys = list(self.memory.keys())
            for key in keys:
                if self.memory[key][1] > 30:
                    del self.memory[key]
                    del self.memory1[key]
                    del self.memory_prob[key]
                    continue
                self.memory.update({key: [self.memory[key][0], self.memory[key][1] + 1]})
        return frame, info

    def pose_detection(self, frame):
        h, w, _ = frame.shape
        bbox, label, score, label_id, kpts = self.y7_pose.predict(frame)
        # ************************** CHECK AND REMOVE NOISE SKELETON ****************************
        id_hold = []
        for i, box in enumerate(bbox):
            # check and remove bbox
            if box[0] < 5 or box[1] < 5 or box[2] > w - 5 or box[3] > h - 5:
                id_hold.append(False)
                continue
            id_hold.append(True)
        bbox, score, kpts = np.array(bbox), np.array(score), np.array(kpts)
        bbox, score, kpts = bbox[id_hold], score[id_hold], kpts[id_hold]
        return bbox, score, kpts

    def face_recognition(self, frame):
        face = {}
        info = {}
        h, w, _ = frame.shape
        if self.turn_detect_face:
            # ***************************** FACE DETECTION **********************************
            bbox_f, label_f, label_id_f, score_f, landmark_f = self.face_model.detect(frame)
            for idx, box in enumerate(bbox_f):
                # check and remove face small
                if box[2] - box[0] < 15 or box[3] - box[1] < 15:
                    continue
                # *************************** FACE RECOGNITION ******************************
                feet = self.face_model.face_encoding(frame, kps=np.array(landmark_f[idx]))
                data = self.face_model.face_compare(feet, threshold=0.3)
                name = data[1]
                face.update({name: landmark_f[idx]})
                info.update({name: [data[0], data[2]]})
                draw_result(frame, box, '', score_f[idx], landmark_f[idx])
            self.turn_detect_face = False
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 0), 2)
        return face, frame, info

    def draw_frame(self, frame, box, action, name, track_id):
        color = (0, 255, 255)
        color1 = (255, 255, 0)
        try:
            if action[0] == "Fall Down":
                color = (0, 0, 255)
            cv2.putText(frame, '{}: {}'.format(name, track_id),
                        (max(box[0] - 20, 0), box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color1, 2, cv2.LINE_AA)
            cv2.putText(frame, '{}'.format(action[0]),
                        (max(box[0] - 20, 0), box[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        except:
            cv2.putText(frame, '{}: {}'.format(name, track_id),
                        (max(box[0] - 20, 0), box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color1, 2, cv2.LINE_AA)
            cv2.putText(frame, '{}'.format('Pending ...'),
                        (max(box[0] - 20, 0), box[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def compute_distance(nose_body, nose_face):
        """
        function: compute distance between nose pose body and nose kpt face
        """
        nose_face = nose_face.astype('float')
        distance = np.sqrt(np.sum((nose_face - nose_body) ** 2, axis=1))
        d_min = np.amin(distance)
        idx = np.argmin(distance)
        return d_min, idx


