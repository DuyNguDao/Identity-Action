import numpy as np
import torch
import sys
from os.path import exists as file_exists, join
from track_sort.sort.detection import Detection
from track_sort.sort.tracker import Tracker


class SORT(object):
    def __init__(self, max_iou_distance=0.7, max_age=70, n_init=3):
        self.tracker = Tracker(max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, kpts, img_org):

        self.height, self.width = img_org.shape[:2]
        bbox_xywh = self._xyxy_to_xywh(bbox_xywh)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, kpts[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        keypoints = np.array([d.kpt for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, confidences, kpts)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            conf = track.conf
            kpt = track.kpt
            pts = np.array(track.keypoints_list, dtype=np.float32)
            outputs.append({'bbox': [x1, y1, x2, y2], 'score': conf, 'kpt': kpt, 'id': track_id, 'list_kpt': pts})
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh
    @staticmethod
    def _xyxy_to_xywh(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_xywh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_xywh = bbox_xyxy.clone()
        bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        bbox_xywh[:, 0] = bbox_xyxy[:, 0] + bbox_xywh[:, 2] // 2
        bbox_xywh[:, 1] = bbox_xyxy[:, 1] + bbox_xywh[:, 3] // 2

        return bbox_xywh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

