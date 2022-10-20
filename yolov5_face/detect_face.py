# -*- coding: UTF-8 -*-
# Dev: Duy Ngu Dao
# Organization: Da Nang university of science and technology
# Time: 9/5/2022


# ----------------- IMPORT LIBRARY -------------------------------
import cv2
import torch
import copy
from yolov5_face.models.experimental import attempt_load
from yolov5_face.utils.datasets import letterbox
from yolov5_face.utils.general import check_img_size, non_max_suppression_face, scale_coords
import sys
from pathlib import Path
import numpy as np
import pickle
from yolov5_face.models.yolo import Model

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# fix change name folder
sys.path.insert(0, str(ROOT))


class Y5DetectFace:
    def __init__(self, weights):
        """
        params weights: yolov5_face.pt
        """
        self.weights = str(weights)
        self.model_image_size = 800
        self.conf_threshold = 0.5
        self.iou_threshhold = 0.45
        with torch.no_grad():
            self.model, self.device = self.load_model(use_cuda=True)
            self.model.to(device=self.device)
            self.model.eval()
            self.class_names = self.model.module.names if hasattr(self.model, "module") else self.model.names

    def load_model(self, use_cuda=False):
        if use_cuda:
            use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else 'cpu')
        from yolov5_face.models.yolo import Model
        model = Model(cfg=ROOT / 'models/yolov5s.yaml', nc=1)
        model.load_state_dict(torch.load(self.weights, map_location=device))
        # model = attempt_load(self.weights, map_location=device)
        return model, device

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        # clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords

    def preprocess_image(self, image_rgb):
        orgimg = image_rgb
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = self.model_image_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.model_image_size, s=self.model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, image_rgb):
        img = self.preprocess_image(image_rgb)
        pred = self.model(img)[0]
        # NMS
        pred = non_max_suppression_face(pred, self.conf_threshold, self.iou_threshhold)
        bboxes = []
        labels = []
        labels_id = []
        scores = []
        landmarks = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], image_rgb.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].detach().cpu().view(-1).tolist()
                    bboxes.append(list(map(int, xyxy)))
                    conf = det[j, 4].detach().cpu().numpy()
                    scores.append(float(conf))
                    landmark = det[j, 5:15].detach().cpu().view(-1).tolist()
                    landmark = np.array(landmark, dtype='int32').reshape((5, 2)).tolist()
                    landmarks.append(landmark)
                    class_num = det[j, 15].detach().cpu().numpy()
                    labels.append(self.class_names[int(class_num)])
                    labels_id.append(int(class_num))
        return bboxes, labels, labels_id, scores, landmarks


def draw_result(image, boxes, label, scores, landmarks):
    color = (0, 255, 0)
    xmin, ymin, xmax, ymax = boxes
    h, w, c = image.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=tl, lineType=cv2.LINE_AA)
    tf = max(tl - 1, 1)  # font thickness
    # label = str(scores)[:5]
    cv2.putText(image, label, (xmin, ymin - 2), 0, tl, [225, 0, 0], thickness=tf*3, lineType=cv2.LINE_AA)
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i, kps in enumerate(landmarks):
        point_x = kps[0]
        point_y = kps[1]
        cv2.circle(image, (point_x, point_y), tl+1, clors[i], -1)
    return image


if __name__ == '__main__':
    path_models = 'weights/yolov5s-face.pt'
    path_image = '/home/duyngu/Downloads/video_test/test_face.jpg'
    y5_face = Y5DetectFace(weights=path_models)

    # print(model)
    # torch.save(y5_face.model.state_dict(), 'yolov5_face.pt')
    # image = cv2.imread(path_image)
    # bbox, label, score, landmark = y5_face.predict(image[:, :, ::-1])
    # for id, box in enumerate(bbox):
    #     image = draw_result(image, box, label[id], score[id], landmark[id])
    # cv2.imshow('result', image)
    # cv2.waitKey(0)
