import os
import torch
import numpy as np

import sys
sys.path.append('./Actionsrecognition')
from classification_stgcn.Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from classification_stgcn.Actionsrecognition.pose_utils import normalize_points_with_size, scale_pose
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self, skip=False, device='cpu'):
        if skip:
            weight_file = ROOT / 'weights/tsstg-model_skip.pth'
        else:
            weight_file = ROOT / 'weights/tsstg-model.pth'

        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Standing', 'Stand up', 'Sitting', 'Sit down', 'Lying Down', 'Walking', 'Fall Down']
        self.num_class = len(self.class_names)
        if device == 'cpu':
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Model ST-GCN: {}, device: {}".format(str(weight_file).split("/")[-1], self.device))
        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        # pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)
        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]
        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        mot = mot.to(self.device)
        pts = pts.to(self.device)
        out = self.model((pts, mot)).detach().cpu().numpy()
        label = self.class_names[out[0].argmax()]
        score = out[0].max()*100
        return [label], score
