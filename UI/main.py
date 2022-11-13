# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/duyngu/Desktop/Project_Graduate/UI/main.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1536, 864)
        Form.setMinimumSize(QtCore.QSize(1536, 864))
        Form.setMaximumSize(QtCore.QSize(1920, 1080))
        Form.setStyleSheet("background-color: rgb(24, 23, 61)")
        # Form.setStyleSheet("background-image: url(icon/dh-bach-khoa-da-nang.jpg);")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame = QtWidgets.QFrame(Form)

        self.frame.setMinimumSize(QtCore.QSize(1536, 864))
        self.frame.setMaximumSize(QtCore.QSize(1920, 1080))
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setMinimumSize(QtCore.QSize(1366, 200))
        self.frame_4.setMaximumSize(QtCore.QSize(1920, 200))
        self.frame_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.logo_dhbk = QtWidgets.QLabel(self.frame_4)
        self.logo_dhbk.setMinimumSize(QtCore.QSize(200, 200))
        self.logo_dhbk.setMaximumSize(QtCore.QSize(200, 200))
        self.logo_dhbk.setStyleSheet("background-color: rgb(136, 138, 133);")
        self.logo_dhbk.setObjectName("logo_dhbk")
        self.horizontalLayout_3.addWidget(self.logo_dhbk)
        self.label_2 = QtWidgets.QLabel(self.frame_4)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.logo_dtvt = QtWidgets.QLabel(self.frame_4)
        self.logo_dtvt.setMinimumSize(QtCore.QSize(200, 200))
        self.logo_dtvt.setMaximumSize(QtCore.QSize(200, 200))
        self.logo_dtvt.setStyleSheet("background-color: rgb(136, 138, 133);")
        self.logo_dtvt.setObjectName("logo_dtvt")
        self.horizontalLayout_3.addWidget(self.logo_dtvt)
        self.verticalLayout_2.addWidget(self.frame_4)
        self.frame_6 = QtWidgets.QFrame(self.frame)
        self.frame_6.setMinimumSize(QtCore.QSize(1366, 664))
        self.frame_6.setMaximumSize(QtCore.QSize(1920, 880))
        self.frame_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_4.setContentsMargins(0, 0, 20, 0)
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_5 = QtWidgets.QFrame(self.frame_6)
        self.frame_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.frame_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMinimumSize(QtCore.QSize(500, 768))
        self.frame_3.setMaximumSize(QtCore.QSize(500, 768))
        self.frame_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_3.setContentsMargins(20, 20, 20, 20)
        self.verticalLayout_3.setSpacing(10)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_7 = QtWidgets.QFrame(self.frame_3)
        self.frame_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.image_action = QtWidgets.QLabel(self.frame_7)
        self.image_action.setMinimumSize(QtCore.QSize(200, 200))
        self.image_action.setMaximumSize(QtCore.QSize(200, 200))
        self.image_action.setStyleSheet("background-color: rgb(136, 138, 133);\n"
"border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.image_action.setAlignment(QtCore.Qt.AlignCenter)
        self.image_action.setWordWrap(False)
        self.image_action.setObjectName("image_action")
        self.verticalLayout_4.addWidget(self.image_action)
        self.verticalLayout_3.addWidget(self.frame_7, 0, QtCore.Qt.AlignHCenter)
        self.formLayout_7 = QtWidgets.QFormLayout()
        self.formLayout_7.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.formLayout_7.setFormAlignment(QtCore.Qt.AlignCenter)
        self.formLayout_7.setSpacing(10)
        self.formLayout_7.setObjectName("formLayout_7")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setMinimumSize(QtCore.QSize(90, 30))
        self.label_3.setMaximumSize(QtCore.QSize(90, 30))
        self.label_3.setWhatsThis("")
        self.label_3.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"border-style:inset;")
        self.label_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_3.setObjectName("label_3")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.name_people = QtWidgets.QLabel(self.frame_3)
        self.name_people.setMinimumSize(QtCore.QSize(210, 30))
        self.name_people.setMaximumSize(QtCore.QSize(210, 30))
        self.name_people.setStyleSheet("border-color: rgb(255, 255, 255);\n"
"background-color: rgb(136, 138, 133);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.name_people.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.name_people.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.name_people.setTextFormat(QtCore.Qt.AutoText)
        self.name_people.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.name_people.setObjectName("name_people")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.name_people)
        self.label_4 = QtWidgets.QLabel(self.frame_3)
        self.label_4.setMinimumSize(QtCore.QSize(90, 30))
        self.label_4.setMaximumSize(QtCore.QSize(90, 30))
        self.label_4.setWhatsThis("")
        self.label_4.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"border-style:inset;")
        self.label_4.setObjectName("label_4")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.name_action = QtWidgets.QLabel(self.frame_3)
        self.name_action.setMinimumSize(QtCore.QSize(210, 30))
        self.name_action.setMaximumSize(QtCore.QSize(210, 30))
        self.name_action.setStyleSheet("border-color: rgb(255, 255, 255);\n"
"background-color: rgb(136, 138, 133);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.name_action.setObjectName("name_action")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.name_action)
        self.label_6 = QtWidgets.QLabel(self.frame_3)
        self.label_6.setMinimumSize(QtCore.QSize(90, 30))
        self.label_6.setMaximumSize(QtCore.QSize(90, 30))
        self.label_6.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"border-style:inset;")
        self.label_6.setObjectName("label_6")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.time_action = QtWidgets.QLabel(self.frame_3)
        self.time_action.setMinimumSize(QtCore.QSize(210, 30))
        self.time_action.setMaximumSize(QtCore.QSize(210, 30))
        self.time_action.setStyleSheet("border-color: rgb(255, 255, 255);\n"
"background-color: rgb(136, 138, 133);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.time_action.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.time_action.setObjectName("time_action")
        self.formLayout_7.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.time_action)
        self.verticalLayout_3.addLayout(self.formLayout_7)
        self.frame_2 = QtWidgets.QFrame(self.frame_3)
        self.frame_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setContentsMargins(20, 20, 5, 20)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.rtsp_video = QtWidgets.QLineEdit(self.frame_2)
        self.rtsp_video.setMinimumSize(QtCore.QSize(300, 30))
        self.rtsp_video.setMaximumSize(QtCore.QSize(300, 30))
        self.rtsp_video.setStyleSheet("background-color: rgb(136, 138, 133);\n"
"border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.rtsp_video.setObjectName("rtsp_video")
        self.horizontalLayout_2.addWidget(self.rtsp_video)
        self.verticalLayout_3.addWidget(self.frame_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(75, -1, 75, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.run_button = QtWidgets.QPushButton(self.frame_3)
        self.run_button.setMinimumSize(QtCore.QSize(100, 100))
        self.run_button.setMaximumSize(QtCore.QSize(100, 100))
        self.run_button.setStyleSheet("QPushButton{\n"
"    \n"
"border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"background-color: rgb(136, 138, 133);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: rgb(32, 74, 135);\n"
"\n"
"}")
        self.run_button.setText("")
        self.run_button.setIconSize(QtCore.QSize(50, 50))
        self.run_button.setObjectName("run_button")
        self.horizontalLayout.addWidget(self.run_button)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.stop_button = QtWidgets.QPushButton(self.frame_3)
        self.stop_button.setMinimumSize(QtCore.QSize(100, 100))
        self.stop_button.setMaximumSize(QtCore.QSize(100, 100))
        self.stop_button.setStyleSheet("QPushButton{\n"
"    \n"
"border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"background-color: rgb(136, 138, 133);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: rgb(32, 74, 135);\n"
"\n"
"}")
        self.stop_button.setText("")
        self.stop_button.setIconSize(QtCore.QSize(50, 50))
        self.stop_button.setObjectName("stop_button")
        self.horizontalLayout.addWidget(self.stop_button, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout.addWidget(self.frame_3, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.horizontalLayout_4.addWidget(self.frame_5)
        self.label_screen = QtWidgets.QLabel(self.frame_6)
        self.label_screen.setMinimumSize(QtCore.QSize(1366, 768))
        self.label_screen.setMaximumSize(QtCore.QSize(1366, 768))
        self.label_screen.setStyleSheet("background-color: rgb(136, 138, 133);\n"
"border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.label_screen.setAlignment(QtCore.Qt.AlignCenter)
        self.label_screen.setObjectName("label_screen")
        self.horizontalLayout_4.addWidget(self.label_screen)
        self.verticalLayout_2.addWidget(self.frame_6)
        self.verticalLayout_5.addWidget(self.frame)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.logo_dhbk.setText(_translate("Form", "logo dhbk"))
        self.label_2.setWhatsThis(_translate("Form", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">RTSP VIDEO: </span></p></body></html>"))
        self.label_2.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600; color:#ef2929;\">IDENTITY RECOGNITION AND APPLICATION ACTIONS FOR FALL DETECTION IN THE ELDERLY</span></p><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ef2929;\">Student 1: DAO DUY NGU Code: 106180036</span></p><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ef2929;\">Student2: LE VAN THIEN Code: 106180051</span></p><p align=\"center\"><span style=\" font-size:16pt; font-weight:600; color:#ef2929;\">Instructor: PhD. TRAN THI MINH HANH</span></p><p><br/></p></body></html>"))
        self.logo_dtvt.setText(_translate("Form", "logo dtvt"))
        self.image_action.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">Fall Action</span></p></body></html>"))
        self.label_3.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">Name</span></p></body></html>"))
        self.name_people.setText(_translate("Form", "None"))
        self.label_4.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">Action</span></p></body></html>"))
        self.name_action.setText(_translate("Form", "None"))
        self.label_6.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ffffff;\">Time</span></p></body></html>"))
        self.time_action.setText(_translate("Form", "None"))
        self.label.setWhatsThis(_translate("Form", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">RTSP VIDEO: </span></p></body></html>"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">RTSP VIDEO:</span></p></body></html>"))
        self.label_screen.setText(_translate("Form", "Camera"))