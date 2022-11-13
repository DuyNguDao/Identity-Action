"""
IDENTITY RECOGNITION AND APPLICATION ACTIONS FOR FALL DETECTION IN THE ELDERLY
Member: DAO DUY NGU, LE VAN THIEN
Mentor: PhD. TRAN THI MINH HANH
Time: 12/11/2022
contact: ddngu0110@gmail.com, ngocthien3920@gmail.com
"""

import cv2
from human_action_and_identity import ActionAndIdentityRecognition
import time


def run(url=None, flag_save=False, fps=30, name_video="recording_action.avi"):
    assert url
    cap = cv2.VideoCapture(url)
    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_height, frame_width)
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
    model = ActionAndIdentityRecognition()
    skip = True
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        if h > h_norm or w > w_norm:
            rate_max = max(h_norm / h, w_norm / w)
            frame = cv2.resize(frame, (int(rate_max * w), int(rate_max * h)), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        frame, info = model.processing(frame, skip)
        # ******************************************** SKIP ONE FRAME *********************************
        skip = not skip
        # ******************************************** SHOW *******************************************
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        # ******************************************** SAVE VIDEO *************************************
        if flag_save is True:
            video_writer.write(frame)
    cap.release()
    if flag_save is True:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    url = '/home/duyngu/Downloads/video_test/video_hanh_dong_truong_hoc.mp4'
    # url = '/home/duyngu/Downloads/video_test/video2.avi'
    # url = '/home/duyngu/Downloads/video_test/20221001153334758_7F01683RAZE9C1D.mp4'
    cv2.namedWindow('video')
    run(url)
