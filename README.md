# IDENTITY AND APPLICATION ACTIONS FOR FALL DETECTION IN THE ELDERLY
## Introduction
This is project of identity recognition and human action that is widely applied in life such as: falls in the elderly, stroke patients or behavior recognition in children
## video demo
![](./information/recog_recording.gif)
## System Diagram

![](./information/System_diagram.png)
## Dev
```
Member:
 - DAO DUY NGU
 - LE VAN THIEN
Instructor: TRAN THI MINH HANH
```
## Usage
### Install package
```
git clone https://github.com/DuyNguDao/Project_Graduate.git
```
```
cd Project_Graduate
```
```
conda create --name human_action python=3.8
```
```
pip install -r requirements.txt
```
### Download
model yolov7 pose state dict:
[yolov7_w6_pose](https://drive.google.com/file/d/1z8WVeqbjUKeibV0ZRDL5tBac9Ry8AkB3/view?usp=sharing)
### Quick start
#### start and config url
```
python run_video.py
```
#### start with terminal
```
python detect_video.py --fn <url if is video or 0>
```

## Datasets and result model training
### Dataset human action
[Human action](https://www.kaggle.com/datasets/ngoduy/dataset-video-for-human-action-recognition)
### Dataset Face Detection
[Face detection](https://www.kaggle.com/datasets/ngoduy/dataset-for-face-detection)
### Dataset Face Recognition
[Face recognition](https://www.kaggle.com/datasets/ngoduy/face-recognition-with-webface-and-facescrib)

## Result face recognition
### Diagram

![](information/face_recognition_diagram.png)
### Face detection
#### FDDB DATA
* Confusion matrix of YOLO5Face

![](./information/confusion_matrix_FDDB_YOLO.png)
* Confusion matrix of RetinaFace

![](./information/confusion_matrix_FDDB_RetinaFace.png)

#### WIDERFACE Val
* Confusion matrix of YOLO5Face

![](./information/confusion_matrix_WIDERFACE_YOLO.png)
* Confusion matrix of RetinaFace

![](./information/confusion_matrix_WIDERFACE_RetinaFace.png)
#### Result compare: Accuracy, Precision, Recall, Time processing
Config Computer:
+ CPU: AMD Ryzen 7 4800H với 16G RAM DDR4
+ GPU: NVIDIA GeForce GTX 1650 với 4G RAM DDR6

![](./information/table_compare_face_detect.png)

#### Face landmark loss

![](./information/loss_nme.png)
### Face recognition
#### Backbone compare:
* MobileFaceNet

![](./information/MobileFaceNet.png)
* ResNet18

![](information/ResNet18.png)
#### Methods compare: LOOCV (Leave One Of Cross Validation), Time processing with FaceScrub data

![](./information/table_compare_face_recog.png)
### Diagram Accuracy Thresh

![](./information/AT.png)
## Result human action pose
### Diagram

![](./information/human_action_diagram.png)
### Confusion matrix
* Model LSTM (Long Short Term Memory)

![](./information/confusion_matrix_LSTM.png)
* Model ST-GCN (Spatial Temporal - Graph Convolutional Network)

![](./information/confusion_matrix_STGCN.png)
### Methods compare
Accuracy, Precision, Recall, F1-score, Time processing

Config Computer:
+ CPU: AMD Ryzen 7 4800H với 16G RAM DDR4
+ GPU: NVIDIA GeForce GTX 1650 với 4G RAM DDR6

![](./information/table_compare_action.png)

### Compare ST-GCN + YOLOv7-Pose and ST-GCN + YOLOv3 + Alphapose
Config Computer:
+ CPU: Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz
+ GPU: GEFORCE RTX 1050 với RAM 4G

![](./information/table_compare_2_method.png)

+ Confusion matrix of ST-GCN with skeleton data export from yolov3 + alphapose

![](./information/confusion_matrix_yolov3.png)

### Training
#### Human action
* https://github.com/DuyNguDao/LSTM.git
* https://github.com/DuyNguDao/ST-GCN-Pytorch.git
#### Face Detection
* https://github.com/deepcam-cn/yolov5-face.git
* https://github.com/biubug6/Pytorch_Retinaface.git
#### Face recognition
* https://github.com/DuyNguDao/ArcFace_Pytorch.git

## Contact
```
Address: Da Nang University of Science and Technology
Email: ddngu0110@gmail.com, ngocthien3920@gmail.com
```
## Acknowledgements
* https://github.com/deepinsight/insightface.git
* https://github.com/deepcam-cn/yolov5-face.git
* https://github.com/WongKinYiu/yolov7.git
* https://github.com/biubug6/Pytorch_Retinaface.git
* https://github.com/GajuuzZ/Human-Falling-Detect-Tracks.git



