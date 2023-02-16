import PySimpleGUI as sg
import os.path
import io
from mgrUtils import *
import base64
from PIL import Image
import numpy as np
import cv2

file_list_column = [
    [sg.Text("Enter URL for youtube video")],
    [sg.Text('URL', size =(4, 1)), sg.InputText("https://youtu.be/2CIxM7x-Clc")],
    [sg.Button("Load Video WEB")],
    [sg.HSeparator()],

    [sg.Text("Enterpath for local video")],
    [sg.Text('PATH', size =(4, 1)), sg.InputText(key='-PATH-', default_text=R"inputData\baseLaneCrossing.avi")],
    [sg.Button("Load_Video_FILE")],

    [sg.HSeparator()],
    [sg.Text('Model Path', size =(10, 1)), sg.InputText("models/tusimple_18.pth", key="-MODEL PATH-")],
    [sg.Text('Model Type', size =(10, 1)), sg.Radio("ModelType.TUSIMPLE", "MODEL_TYPE", default=True, key="-TUSIMPLE-")],
    [sg.Text('Model Type', size =(10, 1)), sg.Radio("ModelType.CULANE", "MODEL_TYPE", default=False, key="-CULANE-")],
    [sg.Button("Init Model")],
    [sg.HSeparator()],
    [sg.Text('Output filename', size =(20, 1)), sg.InputText("output.mp4", key="-OUTPUT FILENAME-")],
    [sg.Button("RUN FILE")],
    [sg.Button("RUN SCREEN")],
]


image_viewer_column = [
    [sg.Text("Current Status")],
    [sg.Text(size=(40, 1), key="-STATUS-")],
    [sg.Image(key="-IMAGE-")],
]


layout = [
    [
        sg.Column(file_list_column),
        sg.VSeparator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Video model runner", layout)
capture = ""
laneDetector = ""

def runLaneDetectionOnScreen(capture, laneDetector):
    while capture.isOpened():
        try:
            ret, frame = capture.read()
        except:
            print("Exception at reading capture, continuing")
            continue

        if ret:
            outputImage = laneDetector.detect_lanes(frame)
            imgBytes = cv2.imencode('.ppm', outputImage)[1].tobytes()
            window['-IMAGE-'].update(data=imgBytes)
            window.refresh()
            
        else:
            break


while True:
    event, values = window.read()

    if event == "Load Video WEB":
        window["-STATUS-"].update("Loading Video.....")
        capture = loadVideoToStreanm(values[0])
        window["-STATUS-"].update("Video loaded")
    
    if event == "Load_Video_FILE":
        window["-STATUS-"].update("Loading Video.....")
        capture = loadVideoFromFileToStream(values['-PATH-'])
        window["-STATUS-"].update("Video loaded")

    if event == "Init Model":
        window["-STATUS-"].update("Loading Model....")
        laneDetector = initLaneDetector(values["-MODEL PATH-"], values["-TUSIMPLE-"], True)
        window["-STATUS-"].update("Model Loaded!")

    if event == "RUN FILE":
        window["-STATUS-"].update("Generating output....")
        runLaneDetectionToFile(capture, laneDetector, values["-OUTPUT FILENAME-"])
        window["-STATUS-"].update("Generating output COMPLETED")

    if event == "RUN SCREEN":
        window["-STATUS-"].update("Generating output on the screen")
        runLaneDetectionOnScreen(capture, laneDetector)
        window["-STATUS-"].update("Generating output on the screen COMPLETED")

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

window.close()



