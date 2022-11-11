import torch
import torch.nn as nn
from classification_lstm.models.rnn import RNN
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class Model:
    def __init__(self, skip=True, device='cpu'):

        self.class_names = ['Sit down', 'Lying Down', 'Walking', 'Stand up', 'Standing', 'Fall Down', 'Sitting']
        # self.class_names = ['Fall Down', 'Other action']
        # self.class_names = ['Siting', 'Lying Down', 'Walking or Standing', 'Fall Down']
        # config device cuda or cpu
        if device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RNN(input_size=26, num_classes=len(self.class_names), device=self.device).to(self.device)
        if skip:
            self.path = str(ROOT / 'weights/best_skip_lstm.pt')
        else:
            self.path = str(ROOT / 'weights/best.pt')
        self.load_model()
        self.model.eval()
        print("Model detect face: {}, device: {}".format(self.path.split('/')[-1], self.device))
        # self.class_names = ['Standing', 'Stand up', 'Sitting', 'Sit down', 'Lying Down', 'Walking', 'Fall Down']


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
        pose = pose[:, :, :, :2]
        pose = np.concatenate([pose[:, :, 0:1, :], pose[:, :, 5:, :]], axis=2)  # remove point 1,2,3,4
        # normalize
        pose[:, :, :, 0] /= size_w
        pose[:, :, :, 1] /= size_h
        pose = scale_pose(pose)
        pose = pose.reshape(pose.shape[0], pose.shape[1], pose.shape[2]*pose.shape[3])
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
