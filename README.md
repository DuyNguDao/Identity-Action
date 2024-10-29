# The combination of face identification and action recognition for fall detection
## Introduction
Falls are a very common unexpected accident in the elderly that results in serious injuries such as broken bones, and head injury. Detecting falls and taking fall patients to the emergency room on time is very important. In this project, we propose a method combining face and action recognition for fall detection. Specifically, we identify seven basic actions in elderly daily life based on skeleton data detected using the YOLOv7-Pose model. Two deep models, Spatial-Temporal Graph Convolutional Network (ST-GCN) and Long Short-Term Memory (LSTM), are employed for action recognition on the skeleton data. The experimental results on our dataset show that the ST-GCN model achieved an accuracy of 90% higher than the LSTM model by 7%. 
## video demo
https://user-images.githubusercontent.com/87271954/204276637-f5d343de-9b19-43e4-a34b-5ffb7b696d9a.mp4
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
#### Step 1: Download and setting anaconda environment
```
https://www.anaconda.com/download/success
```
#### Step 2: Open Anaconda Prompt
``` 
Anaconda Prompt on Window
```
#### Step 3: Change directory to project
```
cd Identity-Action
```
#### Step 4: Setup python environment (GPU is required)
```
setup_environment_gpu.bat
```
#### Remove python environment
``` 
remove_env.bat
```

### Download
Model yolov7 pose state-dict: (Note: the model download fee is 20$ through PayPal. Please send information to my email to download. - [ddngu0110@gmail.com]())

[yolov7_w6_pose](https://drive.google.com/file/d/1UiDdOghLoRUOLbgkh41538oEXSG4dDUh/view?usp=share_link)
### Quick start
#### Start and config URL
```
pip uninstall opencv-python-headless
pip install opencv-python==4.5.5.64
python run_video.py
```
#### Start with the terminal
```
pip uninstall opencv-python-headless
pip install opencv-python==4.5.5.64
python detect_video.py --fn <url if is video or 0>
```
### Run App
#### Step 1: Open Anaconda Prompt
``` 
Anaconda Prompt on Window
```
#### Step 2: Change directory to project
```
cd Identity-Action
```
#### Step 3: Run app
```
run_app.bat
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

![](./information/table_compare_face_detect.PNG)

#### Face landmark loss

![](./information/loss_nme.png)
### Face recognition
#### Backbone compare:
* MobileFaceNet

![](./information/MobileFaceNet.png)
* ResNet18

![](information/ResNet18.png)
#### Methods compare: LOOCV (Leave-One-Out Cross-Validation), Time processing with FaceScrub data

![](./information/table_compare_face_recog.PNG)
### Diagram Accuracy Thresh

![](./information/AT.png)
## Result human action pose
### Diagram

![](./information/human_action_diagram.png)
### Backbone
#### LSTM

![](./information/lstm.PNG)

### ST-GCN

![](./information/stgcn.PNG)

### Two Stream ST-GCN

![](./information/two-stgcn.PNG)
### Confusion matrix
* Model LSTM (Long Short Term Memory)

![](./information/confusion_matrix_LSTM.png)
* Model ST-GCN (Spatial Temporal - Graph Convolutional Network)

![](./information/confusion_matrix_STGCN.png)
### Methods compare
Accuracy, Precision, Recall, F1-score, Time processing

Config Computer:
+ CPU: AMD Ryzen 7 4800H with 16G RAM DDR4
+ GPU: NVIDIA GeForce GTX 1650 with 4G RAM DDR6

![](./information/table_compare_action.png)

### Compare ST-GCN + YOLOv7-Pose and ST-GCN + YOLOv3 + Alphapose
Config Computer:
+ CPU: AMD Ryzen 7 4800H với 16G RAM DDR4
+ GPU: NVIDIA GeForce GTX 1650 với 4G RAM DDR6

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
## Citation
```
@article{
  title={The combination of face identification and action recognition for fall detection},
  author={Dao Duy Ngu, Le Van Thien, Tran Thi Minh Hanh, Nguyen Thi Hong Yen, Dao Duy Tuan},
  journal={Journal of Science and Technology, Issue on Information and Communications Technology, ISSN: 1859-1531},
  Pages={37-44, Vol. 20, No. 12.2, 2022}
  year={2022}
}
```
## Contact
```
The University of Da Nang, The University of Science and Technology
Address: 54, Nguyen Luong Bang street, Lien Chieu district, Da Nang City, Viet Nam
```
## Acknowledgements
* https://github.com/deepinsight/insightface.git
* https://github.com/deepcam-cn/yolov5-face.git
* https://github.com/WongKinYiu/yolov7.git
* https://github.com/biubug6/Pytorch_Retinaface.git
* https://github.com/GajuuzZ/Human-Falling-Detect-Tracks.git
* https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git




