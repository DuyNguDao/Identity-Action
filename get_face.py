"""
Module: Record Face
"""
import cv2
import time
from pathlib import Path
from yolov5_face.detect_face import draw_result
import argparse
from face_recognition.face import Face_Model


face_model = Face_Model()
cap = cv2.VideoCapture(0)
name = input("Enter Name: ")
list_image = []
print("\t\t\tSTART RECORD FACE\t\t\t")
state = 'Normal'
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    key = cv2.waitKey(1) & 0xFF
    # create region
    thresh = 200
    xmin, ymin = w//2 - thresh, h//2 - thresh
    xmax, ymax = w//2 + thresh, h//2 + thresh
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, cv2.LINE_AA)
    blur = cv2.blur(frame, (15, 15))
    for i in range(5):
        blur = cv2.blur(blur, (15, 15))
    blur[ymin:ymax, xmin:xmax] = frame[ymin:ymax, xmin:xmax]
    frame = blur
    frame = cv2.putText(frame, f' MENU: Record: R -- Save Data: S -- Clear: P -- New ID: N -- Exit: E',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    if key == ord('r'):
        state = 'Record'
    elif key == ord('s'):
        state = 'Stop'
    elif key == ord('p'):
        state = 'Record Again'
        list_image.clear()
    elif key == ord('n'):
        state = 'New ID'
    elif key == ord('e'):
        break

    if state == 'Record':
        if h > 1080 and w > 1920:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
        bbox, label, label_id, score, landmark = face_model.detect(frame)
        if len(bbox) != 0:
            list_image.append(frame.copy())
        for idx, box in enumerate(bbox):
            draw_result(frame, box, '', score[idx], landmark[idx])
    elif state == 'Stop':
        face_model.create_data(list_image, name)
        state = 'Normal'
    elif state == 'Record again':
        list_image.clear()
        state = 'Normal'
    elif state == 'New ID':
        list_image.clear()
        name = input("Enter Name: ")
        state = 'Normal'

    frame = cv2.putText(frame, f'ID: {name}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame = cv2.putText(frame, f'State: {state}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('video', frame)

cap.release()
cv2.destroyAllWindows()
print('Finish!')
