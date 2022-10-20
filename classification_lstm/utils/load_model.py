import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
from classification_lstm.models.rnn import RNN
import csv
import numpy as np


class Model:
    def __init__(self, path):
        # config device cuda or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RNN(input_size=26, num_classes=7, device=self.device).to(self.device)
        self.path = path
        self.load_model()
        self.model.eval()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number parameters: ", count_parameters(self.model))
        self.class_names = ['Standing', 'Stand up', 'Sitting', 'Sit down', 'Lying Down', 'Walking', 'Fall Down']

    def load_model(self):
        """
        function: load model and parameter
        :return:
        """
        # load model
        self.model.load_state_dict(torch.load(self.path, map_location=self.device))

    def preprocess_data(self, list_data, size_w, size_h):
        """
        function: preprocessing image
        :param image: array image
        :return:
        """
        def compute_angle(features):
            # ********************************* COMPUTE ANGLE *********************************************
            # angle knee right
            knee_hip = features[:, :, 14:15, :] - features[:, :, 12:13, :]
            knee_ankle = features[:, :, 14:15, :] - features[:, :, 16:17, :]
            a = np.sum(knee_hip * knee_ankle, axis=3)
            b = np.sqrt(np.sum(knee_hip ** 2, axis=3)) * np.sqrt(np.sum(knee_ankle ** 2, axis=3))
            b = np.where(b == 0, 1, b)
            angle_knee_right = a / b

            # angle knee left
            knee_hip = features[:, :, 13:14, :] - features[:, :, 11:12, :]
            knee_ankle = features[:, :, 13:14, :] - features[:, :, 15:16, :]
            a = np.sum(knee_hip * knee_ankle, axis=3)
            b = np.sqrt(np.sum(knee_hip ** 2, axis=3)) * np.sqrt(np.sum(knee_ankle ** 2, axis=3))
            b = np.where(b == 0, 1, b)
            angle_knee_left = a / b

            # angle hip right
            hip_shoulder = features[:, :, 12:13, :] - features[:, :, 6:7, :]
            hip_knee = features[:, :, 12:13, :] - features[:, :, 14:15, :]
            a = np.sum(hip_shoulder * hip_knee, axis=3)
            b = np.sqrt(np.sum(hip_shoulder ** 2, axis=3)) * np.sqrt(np.sum(hip_knee ** 2, axis=3))
            b = np.where(b == 0, 1, b)
            angle_hip_right = a / b

            # angle hip left
            hip_shoulder = features[:, :, 11:12, :] - features[:, :, 5:6, :]
            hip_knee = features[:, :, 11:12, :] - features[:, :, 13:14, :]
            a = np.sum(hip_shoulder * hip_knee, axis=3)
            b = np.sqrt(np.sum(hip_shoulder ** 2, axis=3)) * np.sqrt(np.sum(hip_knee ** 2, axis=3))
            b = np.where(b == 0, 1, b)
            angle_hip_left = a / b
            return [angle_hip_left, angle_hip_right, angle_knee_left, angle_knee_right]

        def scale_pose(xy):
            """
            Normalize pose points by scale with max/min value of each pose.
            xy : (frames, parts, xy) or (parts, xy)
            """
            xy_min = np.nanmin(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
            xy_max = np.nanmax(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
            xy = (xy - xy_min) / (xy_max - xy_min) * 2 - 1
            return xy

        pose = np.array(list_data)
        angle = compute_angle(pose)
        angle = []
        pose = np.concatenate([pose[:, :, 0:1, :], pose[:, :, 5:, :]], axis=2)  # remove point 1,2,3,4
        # normalize
        pose[:, :, :, 0] /= size_w
        pose[:, :, :, 1] /= size_h
        pose = scale_pose(pose)
        pose = pose.reshape(pose.shape[0], pose.shape[1], pose.shape[2]*pose.shape[3])
        pose = [pose]
        pose.extend(angle)
        pose = np.concatenate(pose, axis=2)
        pose = torch.tensor(pose)
        return pose

    def predict(self, list_data, size_w, size_h, batch_size=5):
        """
        function: predict image
        :param image: array image bgr
        :return: name class predict and list prob predict
        """
        import math
        data_pose = self.preprocess_data(list_data, size_w, size_h)
        label, score = [], []
        for i in range(math.ceil(len(list_data)/batch_size)):
            if (i+1)*batch_size > len(list_data):
                data = data_pose[i*batch_size:(i+1)*batch_size]
            else:
                data = data_pose[i*batch_size:len(list_data)]
            # data = torch.stack(data)
            data = data.to(self.device)
            out = self.model(data)
            torch.cuda.reset_peak_memory_stats()
            # find max
            _, index = torch.max(out, 1)
            # find prob use activation softmax
            percentage = (nn.functional.softmax(out, dim=1) * 100).tolist()
            for idx, name in enumerate(index):
                label.append(self.class_names[name])
                score.append(max(percentage[idx]))
        return label, score


if __name__ == '__main__':
    import random
    from glob import glob
    import time




