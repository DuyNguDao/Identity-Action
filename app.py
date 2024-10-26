"""
The combination of face identification and action recognition for fall detection
Member: DAO DUY NGU, LE VAN THIEN
Mentor: PhD. TRAN THI MINH HANH
Time: 12/11/2022
contact: ddngu0110@gmail.com, ngocthien3920@gmail.com
"""
import threading
import cv2
import time
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, \
    QVBoxLayout, QMessageBox, QTableWidgetItem, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from datetime import datetime
from UI.main import Ui_Form as Tab_1
from UI.add_data import Ui_Form as Tab_2
from UI.show_database import Ui_Form as Show_database
from UI.show_history import Ui_Form as Show_history
from human_action_and_identity import ActionAndIdentityRecognition
from yolov5_face.detect_face import draw_result
from database.interface_sql import *

# ************************************ RESET EMPTY CACHE CUDA ********************************************
import torch, gc
gc.collect()
torch.cuda.empty_cache()
# ********************************************************************************************************

# *********************************** CONFIG DATA ********************************************************

use_camera = 1
camera_id = 0
url = 0
thread_1_running = False
width = 1536
height = 864

# LOAD MODEL
model_action = ActionAndIdentityRecognition()

# *********************************************************************************************************


def norm_size(w, h):
    return int(w*width), int(h*height)


class ActionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)  # send image signal
    change_information_signal = pyqtSignal(dict)   # send detected information

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.model = model_action

    def run(self):
        global use_camera, url
        self.cap = cv2.VideoCapture(url)
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        h_norm, w_norm = 720, 1280
        if frame_width > w_norm:
            rate = w_norm / frame_width
            h_norm = int(frame_height*rate)
            frame_height = h_norm
            frame_width = w_norm
        skip = True
        video_writer = cv2.VideoWriter('video_demo.avi',
                                       cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))
        while self._run_flag and use_camera == 1:
            start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            # convert size to 1280 - x
            if True:  # w > w_norm: 
                rate = w_norm / w
                frame = cv2.resize(frame, (int(rate * w), int(rate * h)), interpolation=cv2.INTER_AREA)
                h, w, _ = frame.shape
            frame, info = self.model.processing(frame, skip)
            # skip = not skip
            if skip:
                fps = int(1 / (time.time() - start))*2
            skip = not skip
            now = datetime.now()
            cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, now.strftime('%a %H:%M:%S'), (w-120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            video_writer.write(frame)
            self.change_pixmap_signal.emit(frame)
            self.change_information_signal.emit(info)
        self.cap.release()
        self.stop()

    def stop(self):
        self._run_flag = False
        self.cap.release()
        self._run_flag = True


class AddFaceThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        global use_camera
        self.cap = cv2.VideoCapture(0)
        w_norm, h_norm = 1280, 720
        while self._run_flag and use_camera == 2:
            ret, frame = self.cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            # resize 1280, x
            rate = w_norm / w
            frame = cv2.resize(frame, (int(rate * w), int(rate * h)), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape
            self.change_pixmap_signal.emit(frame)
        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.cap.release()
        self._run_flag = True


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Action Management Software'
        self.setWindowIcon(QIcon('icon/logo.png'))
        self.left = 0
        self.top = 0
        self.width = width
        self.height = height
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.table_widget = MyTableWidget(self)
        self.table_widget.tabs.currentChanged.connect(self.on_click)
        self.setCentralWidget(self.table_widget)
        self.show()

    def on_click(self):
        global use_camera, thread_1_running
        if self.table_widget.tabs.currentIndex() == 0:
            thread_1_running = False
            use_camera = 1
        else:
            use_camera = 2
            # self.table_widget.tab2.run()


class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = Camera()
        self.tab2 = AddPeople()
        # self.tab3 = ShowDatabase()
        # Add tabs
        self.tabs.addTab(self.tab1, "Camera")
        self.tabs.addTab(self.tab2, "Add Face")
        # self.tabs.addTab(self.tab3, "Show database")
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


class Camera(QWidget, Tab_1):
    def __init__(self, parent=None):
        super(Camera, self).__init__(parent)
        self.setupUi(self)
        self.screen_width, self.screen_height = 1366, 768
        self.label_screen.resize(self.screen_width, self.screen_height)
        self.width_sub_screen, self.height_sub_sceen = 200, 200
        self.image_action.setPixmap(QtGui.QPixmap("icon/unknown_person.jpg").scaled(200, 200))
        self.run_button.setIcon(QIcon('icon/play.png'))
        self.run_button.clicked.connect(self.run)
        self.stop_button.setIcon(QIcon('icon/pause.png'))
        self.stop_button.clicked.connect(self.stop)
        self.logo_dhbk.setPixmap(QtGui.QPixmap('icon/Logodhbk.jpg').scaled(200, 200))
        self.logo_dtvt.setPixmap(QtGui.QPixmap('icon/logo_dtvt.jpg').scaled(200, 200))
        self.rtsp_video.setText('0')
        self.thread = ActionThread()
        self.thread.change_pixmap_signal.connect(self.update_image_main_screen)
        self.thread.change_information_signal.connect(self.update_data)
        self.show()

    def run(self):
        global use_camera, thread_1_running, url
        if use_camera != 1:
            use_camera = 1
        thread_1_running = False
        if not thread_1_running:

            url = self.rtsp_video.text()
            if len(url) == 0:
                QMessageBox.warning(self, "url not find", "warning")
                self.stop()
            else:
                if url == '0':
                    url = 0  # turn on webcam
                self.thread.start()
            thread_1_running = True

    def stop(self):
        global thread_1_running, use_camera
        try:
            use_camera = 2
            self.thread.stop()
            thread_1_running = False
        except:
            pass

    @pyqtSlot(np.ndarray)
    def update_data(self, data_process):
        try:
            for name in data_process.keys():
                data = data_process[name]
                # now = datetime.now()
                image = data['image']
                qt_img = self.convert_cv_qt(image, self.width_sub_screen, self.height_sub_sceen)
                if data['name'] == 'Unknown':
                    self.image_action.setPixmap(QtGui.QPixmap('icon/unknown_person.jpg').scaled(200, 200))
                else:
                    self.image_action.setPixmap(self.convert_cv_qt(data['image'], 200, 200))
                self.image_action.setPixmap(qt_img)
                self.name_people.setText(data['name'])
                self.name_action.setText(data['action'])
                self.time_action.setText(data['time'])
                # save database
                # add_action(data_tuple=(data['id'], data['name'], data['image'], data['action'], now.strftime('%a %H:%M:%S')),
                #            name_table='action_data')
        except Exception:
            pass

    def update_image_main_screen(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, self.screen_width, self.screen_height)
        self.label_screen.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        # rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class FaceDetectThread(QThread):
    face_info = pyqtSignal(tuple)
    def __init__(self, face_model):
        super().__init__()
        self.face_model = face_model
        self.cv_img = None

    def set_param(self, cv_img):
        self.cv_img = cv_img

    def run(self):
        bbox, label, label_id, score, kpss = self.face_model.detect(self.cv_img)
        self.face_info.emit((bbox, label, label_id, score, kpss))
        self.wait()

class AddPeople(QWidget, Tab_2):
    def __init__(self):
        super(AddPeople, self).__init__()
        self.setupUi(self)
        self.run_button.setIcon(QIcon('icon/play.png'))
        self.run_button.clicked.connect(self.recog)
        self.stop_button.setIcon(QIcon('icon/pause.png'))
        self.stop_button.clicked.connect(self.stop)
        self.save_data.setIcon(QIcon('icon/save.png'))
        self.save_data.clicked.connect(self.save)
        self.logo_dhbk.setPixmap(QtGui.QPixmap('icon/Logodhbk.jpg').scaled(200, 200))
        self.logo_dtvt.setPixmap(QtGui.QPixmap('icon/logo_dtvt.jpg').scaled(200, 200))
        self.button_show_database.clicked.connect(self.show_tab_database)
        self.button_show_history.clicked.connect(self.show_tab_history)
        self.press(None)
        self.face_model = model_action.face_model
        self.recog_flag = False
        self.list_image = []
        self.thread = AddFaceThread()
        #self.thread.change_pixmap_signal.connect(self.update_image_main_screen)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread_face_detect = FaceDetectThread(model_action.face_model)
        self.thread_face_detect.face_info.connect(self.update_image_main_screen)
        self.thread_cap_run = False
        # self.run()
        self.show()

    def run(self):
        global use_camera
        self.thread_cap_run = False
        if use_camera != 2:
            use_camera = 2
        time.sleep(0.5)
        if self.thread_cap_run is False:
            self.thread.start()
            self.thread_cap_run = True

    def press(self, event):
        self.id.clear()
        self.name.clear()

    def show_tab_database(self):
        self.show_tab = ShowDatabase(self.face_model)

    def show_tab_history(self):
        self.show_tab1 = ShowHistory()

    def recog(self):
        if self.name.text().strip() != '':
            ret = QMessageBox.question(self, 'confirm', 'Do you want to recording?\n(Yes) or (No)',
                                       QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                self.list_image = []
                self.run()
                self.recog_flag = True
        else:
            QMessageBox.warning(self, 'Warning!', 'Fill name first')

    def stop(self):
        self.recog_flag = False
        self.thread.stop()

    def save(self):
        try:
            ret = QMessageBox.question(self, 'confirm', "Do you want to save?\n(Yes) or (No)",
                                       QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                self.recog_flag = False
                name = self.name.text()
                id = self.id.text()
                self.id.setText('')
                self.name.setText('')
                t = threading.Thread(target=self.face_model.create_data, args=(self.list_image, name, id))
                t.start()
                # self.face_model.create_data(self.list_image, name, id)
                QMessageBox.information(self, 'Information', 'Completed!')
                self.list_image = []
            else:
                pass
        except:
            QMessageBox.warning(self, 'Warning!', "Can't create data")
            self.list_image = []

    def draw_face(self, frame, box):
        color = (255, 255, 0)
        xmin, ymin, xmax, ymax = box
        h, w, c = frame.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=tl, lineType=cv2.LINE_AA)
        return frame
        
    def update_image_main_screen(self, face_info):
        bbox, label, label_id, score, kpss = face_info
        cv_img = self.image.copy()
        if len(bbox) != 0:
            if self.recog_flag:
                if len(self.list_image) < 200:
                    self.list_image.append(self.image.copy())
        for idx, box in enumerate(bbox):
            cv_img = self.draw_face(cv_img, box)
            cv_img = cv2.resize(cv_img, (1366, 768))
        qt_img = self.convert_cv_qt(cv_img, 1366, 768)
        self.label_screen.setPixmap(qt_img)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.image = cv_img.copy()
        #bbox, label, label_id, score, kpss = self.face_model.detect(cv_img)
        self.thread_face_detect.set_param(cv_img)
        self.thread_face_detect.start()
    
    #def update_image_main_screen(self, cv_img):
    #    """Updates the image_label with a new opencv image"""
    #    self.image = cv_img.copy()
    #   bbox, label, label_id, score, landmark = self.face_model.detect(cv_img)
    #    if len(bbox) != 0:
    #        if self.recog_flag:
    #            if len(self.list_image) < 200:
    #                self.list_image.append(self.image.copy())
    #    for idx, box in enumerate(bbox):
    #        draw_result(cv_img, box, '', score[idx], landmark[idx])
    #    qt_img = self.convert_cv_qt(cv_img, 1366, 768)
    #    self.label_screen.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class ShowDatabase(QWidget, Show_database):
    def __init__(self, face_model):
        super(ShowDatabase, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Database")
        self.delete_database.clicked.connect(self.delete_current)
        self.save_database.clicked.connect(self.save_change)
        self.table_database.horizontalHeader().setDefaultSectionSize(224)
        self.table_database.verticalHeader().setDefaultSectionSize(224)
        self.face_model = face_model
        font = QFont()
        font.setPointSize(15)
        self.table_database.setFont(font)
        self.show_database()
        self.show()

    def show_database(self):
        data_face = get_all_face('faceid')
        self.table_database.setRowCount(len(data_face[0]))
        for index, data in enumerate(data_face[:len(data_face)-1]):
            for idx, header in enumerate(data):
                if index == (len(data_face) - 2):
                    image = QLabel("")
                    image.setScaledContents(True)
                    image.setPixmap(self.convert_cv_qt(header, 224, 224))
                    self.table_database.setCellWidget(idx, index, image)
                else:
                    self.table_database.setItem(idx, index, QTableWidgetItem(str(header)))

    def save_change(self):
        ret = QMessageBox.question(self, 'Warning',
                                   "Are you sure you want to change? \n Ok (Yes) or No (No)",
                                   QMessageBox.No | QMessageBox.Yes)

        if ret == QMessageBox.Yes:
            for row in range(self.table_database.rowCount()):
                fix = []
                for col in range(self.table_database.columnCount()):
                    if col == 2:
                        continue
                    fix.append(self.table_database.item(row, col).text())
                update_info(tuple(fix))
            QMessageBox.information(self, "Save successfully", "Completed.")
        self.show_database()

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def delete_current(self):
        ret = QMessageBox.question(self, 'Warning',
                                   "Are you sure you want to delete? \n Ok (Yes) or No (No)",
                                   QMessageBox.No | QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            x = self.table_database.currentRow()
            code_id = self.table_database.item(x, 0).text()
            delete_face(code_id)
            self.show_database()
            QMessageBox.information(self, "Delete successfully", "Completed.")
            self.face_model.update_data()


class ShowHistory(QWidget, Show_history):
    def __init__(self):
        super(ShowHistory, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Database")
        self.table_database.horizontalHeader().setDefaultSectionSize(224)
        self.table_database.verticalHeader().setDefaultSectionSize(224)
        self.delete_history.clicked.connect(self.delete_all)
        font = QFont()
        font.setPointSize(15)
        self.table_database.setFont(font)
        self.show_history()
        self.show()

    def show_history(self):
        data_face = get_all_action('action_data')
        self.table_database.setRowCount(len(data_face[0]))
        for index, data in enumerate(data_face):
            for idx, header in enumerate(data):
                if index == 2 or index == 4:
                    image = QLabel("")
                    image.setScaledContents(True)
                    image.setPixmap(self.convert_cv_qt(header, 224, 224))
                    self.table_database.setCellWidget(idx, index, image)
                else:
                    self.table_database.setItem(idx, index, QTableWidgetItem(str(header)))

    def delete_all(self):
        ret = QMessageBox.question(self, 'Warning',
                                   "Are you sure you want to delete? \n Ok (Yes) or No (No)",
                                   QMessageBox.No | QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            delete_all_task('action_data')
            self.show_history()
            QMessageBox.information(self, "Delete successfully", "Completed.")

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    del model_action
    torch.cuda.empty_cache()
    sys.exit(app.exec_())
