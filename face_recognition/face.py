import cv2
import numpy as np
import os
import glob
import pickle
from yolov5_face.detect_face import Y5DetectFace, draw_result
from pathlib import Path
from face_recognition.deepface import ArcFacePyTorch, Face
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGTHS = ROOT / 'yolov5_face/weights'


class Face_Model():

    def __init__(self, root_path='face_recognition'):
        self.root_path = root_path
        self.Face_Recognition = ArcFacePyTorch(model_file='face_recognition/Weight/backbone.pth', net='r18', device='gpu')
        self.Face_Detection = Y5DetectFace(weights='yolov5_face/weights/yolov5_face.pt')
        self.database = self.load_data()
        self.data_feet = list(self.database.values())
        self.name_id = list(self.database.keys())

    def detect(self, img):
        return self.Face_Detection.predict(img)

    def create_data(self, list_image, name='unknow'):
        if name == 'unknow':
            name = input("Enter Name")
        feets = []
        for i in tqdm(range(len(list_image)), desc='Processing data'):
            image = list_image[i]
            try:
                bbox, label, label_id, score, kpss = self.Face_Detection.predict(image)
                feet = self.face_encoding(image, kps=np.array(kpss[0]))
                feets.append(feet)
            except:
                continue
        feets = np.sum(np.array(feets), axis=0) / len(feets)
        embed = np.array(feets, dtype='float')
        data = self.load_data()
        data.update({name: embed})
        self.data_feet = list(data.values())
        self.name_id = list(data.keys())
        self.save_data(data)

    def load_data(self):
        path = self.root_path + '/database.pkl'
        if os.path.exists(path): 
            with open(path, 'rb') as f: 
                data = pickle.load(f)
            f.close()
            return data
        else: 
            with open(path, 'wb') as file:
                pickle.dump({}, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()
            with open(path, 'rb') as f:
                data = pickle.load(f)
            file.close()
            return data

    def save_data(self, data):
        path = self.root_path + '/database.pkl'
        if os.path.exists(path):
            with open(path, 'wb') as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    def face_encoding(self, image, kps):
        face_box_class = {'kps':  kps}
        face_box_class = Face(face_box_class)
        feet = self.Face_Recognition.get(image, face_box_class)
        return feet

    def face_compare(self, feet, threshold=0.3):
        try:
            sim_max, idx = self.Face_Recognition.compute_sim(feet, np.array(self.data_feet, dtype='float32'))
            if sim_max > threshold:
                return self.name_id[idx]
            return "Unknown"
        except:
            return "Unknown"


