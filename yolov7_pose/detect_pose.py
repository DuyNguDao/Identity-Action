
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import yaml
from numpy import random
import time
from yolov7_pose.models.experimental import attempt_load
from yolov7_pose.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov7_pose.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7_pose.models.yolo import Model
from yolov7_pose.utils.plots import plot_skeleton_kpts
import time
from pathlib import Path
import sys

# ----------- DON'T TOUCH HERE -------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# fix change name folder
sys.path.insert(0, str(ROOT))
# -----------------------------------------------


class Y7Detect:
    def __init__(self, weights='yolov7_pose/weights/yolov7_w6_pose.pt'):
        """
        params weights: 'yolov7.pt'
        """
        self.weights = weights
        self.model_image_size = 640
        self.conf_threshold = 0.05
        self.iou_threshold = 0.45
        with torch.no_grad():
            self.model, self.device = self.load_model(use_cuda=True)
            print("Model detect pose: {}, device: {}".format(weights.split('/')[-1], self.device))
            self.model.to(device=self.device)
            self.model.eval()
            self.stride = int(self.model.stride.max())  # model stride
            self.image_size = check_img_size(self.model_image_size, s=self.stride)
            self.half = False
            if self.half:
                self.model.half()
            self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

    def load_model(self, use_cuda=False):
        if use_cuda:
            use_cuda = torch.cuda.is_available()
            cudnn.benchmark = True
        device = torch.device("cuda:0" if use_cuda else "cpu")
        with open(ROOT / 'models/hyp.pose.yaml') as f:
            hyp = yaml.safe_load(f)
        model = Model(ROOT / 'models/yolov7_w6_pose.yaml', ch=3, nc=1, anchors=hyp.get('anchors'))
        model.load_state_dict(torch.load(self.weights, map_location=device))
        # model = attempt_load(self.weights, map_location=device)
        print('yolov7 running with {}'.format(device))
        return model, device

    def preprocess_image(self, image_bgr):
        img = letterbox(image_bgr.copy(), self.image_size, stride=self.stride, auto=False)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, image_rgb):
        with torch.no_grad():
            image_rgb_shape = image_rgb.shape
            img = self.preprocess_image(image_rgb)
            pred = self.model(img)[0]

            # apply non_max_suppression
            pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, kpt_label=True,
                                       nc=self.model.yaml['nc'],
                                       nkpt=self.model.yaml['nkpt'])
            bboxes = []
            labels = []
            scores = []
            lables_id = []
            kpts = []
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb_shape, kpt_label=False).round()
                    det[:, 6:] = scale_coords(img.shape[2:], det[:, 6:], image_rgb_shape, kpt_label=True, step=3).round()
                for det_idx, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                    x1 = xyxy[0].detach().cpu().data.numpy()
                    y1 = xyxy[1].detach().cpu().data.numpy()
                    x2 = xyxy[2].detach().cpu().data.numpy()
                    y2 = xyxy[3].detach().cpu().data.numpy()
                    #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                    bboxes.append(list(map(int, [x1, y1, x2, y2])))
                    label = self.class_names[int(cls)]
                    #                        print('[INFO] label: ', label)
                    labels.append(label)
                    lables_id.append(cls.cpu())
                    score = conf.detach().cpu().data.numpy()
                    #                        print('[INFO] score: ', score)
                    scores.append(float(score))
                    kpt = det[det_idx, 6:].detach().cpu().data.numpy()
                    steps = 3
                    num_kpts = len(kpt) // steps
                    point = []
                    for kid in range(num_kpts):
                        x_coord, y_coord = kpt[steps * kid], kpt[steps * kid + 1]
                        if steps == 3:
                            conf_pt = kpt[steps * kid + 2]
                        point.append([int(x_coord), int(y_coord), conf_pt])
                    kpts.append(point)
            return bboxes, labels, scores, lables_id, kpts


def draw_boxes(image, boxes, label=None, scores=None, color=None):
    if color is None:
        color = (0, 255, 0)
    xmin, ymin, xmax, ymax = boxes
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    # cv2.putText(image, label + "-{:d}".format(scores), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image, boxes


def draw_kpts(image, kpts):
    for kpt in kpts:
        line_0 = [4, 2, 0, 1, 3, 5, 6]
        line_1 = [10, 8, 6, 12, 11, 5, 7, 9]
        line_2 = [15, 13, 11, 12, 14, 16]
        cv2.polylines(image, [kpt[line_0]], True, (255, 0, 0), 5)
        cv2.polylines(image, [kpt[line_1]], False, (255, 0, 0), 5)
        cv2.polylines(image, [kpt[line_2]], False, (255, 0, 0), 5)
        for idx, point in enumerate(kpt):
            cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), -1)
    return image


if __name__ == '__main__':
    path_models = '/home/duyngu/Desktop/yolov7-pose/weights/yolov7-w6-pose.pt'

    url = '/home/duyngu/Downloads/video_test/TownCentre.mp4'
    y7_model = Y7Detect(weights=path_models)
    # torch.save(y7_model.model.state_dict(), ROOT / 'weights/yolov7_w6_pose.pt')
    # y7_model.run_video(url)
