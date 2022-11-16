import cv2
import numpy as np
import os
import glob
import pickle
from yolov5_face.detect_face import Y5DetectFace, draw_result
from pathlib import Path
from face_recognition.deepface import ArcFacePyTorch, Face
from tqdm import tqdm
from database.interface_sql import get_all_face, get_all_action, add_face, add_action


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGTHS = ROOT / 'yolov5_face/weights'


class Face_Model():

    def __init__(self, root_path='face_recognition', device='cpu'):
        self.root_path = root_path
        self.Face_Recognition = ArcFacePyTorch(model_file='face_recognition/Weight/backbone.pth', net='r18', device=device)
        self.Face_Detection = Y5DetectFace(weights='yolov5_face/weights/yolov5_face.pt')
        self.database = self.load_data()
        self.data_feet = self.database['embed']
        self.name_id = self.database['name']

    def detect(self, img):
        return self.Face_Detection.predict(img)

    def create_data(self, list_image, name='unknow', id_face='DEV03'):
        feets = []
        face = cv2.imread('icon/unknown_person.jpg')
        face = cv2.resize(face, (112, 112))
        if len(list_image) > 200:
            list_image = list_image[:200]
        for i in tqdm(range(len(list_image)), desc='Processing data'):
            image = list_image[i]
            # try:
            bbox, label, label_id, score, kpss = self.Face_Detection.predict(image)
            feet = self.face_encoding(image, kps=np.array(kpss[0]))
            feets.append(feet)
            if i == 5:
                face = cv2.resize(image[bbox[0][1]:bbox[0][3], bbox[0][0]:bbox[0][2]], (112, 112))
            # except:
            #     continue
        feets = np.sum(np.array(feets), axis=0) / len(feets)
        embed = np.array(feets, dtype='float')
        add_face((id_face, name, face, embed), 'faceid')
        # update data
        self.database = self.load_data()
        self.data_feet = self.database['embed']
        self.name_id = self.database['name']

    def load_data(self):
        id_face, fullname, face, embed = get_all_face('faceid')
        return {'id': id_face, 'name': fullname, 'face': face, 'embed': embed}

    def face_encoding(self, image, kps):
        face_box_class = {'kps':  kps}
        face_box_class = Face(face_box_class)
        feet = self.Face_Recognition.get(image, face_box_class)
        return feet

    def face_compare(self, feet, threshold=0.3):
        try:
            sim_max, idx = self.Face_Recognition.compute_sim(feet, np.array(self.data_feet, dtype='float32'))
            if sim_max > threshold:
                return [self.database['id'][idx], self.name_id[idx], self.database['face'][idx]]
            face = cv2.imread('icon/unknown_person.jpg')
            face = cv2.resize(face, (112, 112))
            return ['None', "Unknown", face]
        except:
            face = cv2.imread('icon/unknown_person.jpg')
            face = cv2.resize(face, (112, 112))
            return ['None', "Unknown", face]

