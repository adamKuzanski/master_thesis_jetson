import pafy
import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

def loadVideoToStreanm(videoUrl):
    print("Trying to load video from: {}".format(videoUrl))
    videoPafy = pafy.new(videoUrl)
    cap = cv2.VideoCapture(videoPafy.streams[-1].url)
    print("VIDEO CAPTURED at {} stream resolution".format(videoPafy.streams[-1]))
    return cap

def loadVideoFromFileToStream(videoPath):
    print("Trying to load video from: {}".format(videoPath))
    cap = cv2.VideoCapture(videoPath)
    print("VIDEO CAPTURED")
    return cap

def initLaneDetector(modelPath, isTuSimple, useGpu):
    modelType = ModelType.TUSIMPLE if isTuSimple else ModelType.CULANE
    print("Setting model with params: \t modelPath: {} , modelType: {} , gpu: {}".format(modelPath, modelType, useGpu))
    return UltrafastLaneDetector(modelPath, modelType, True)

def runLaneDetectionToFile(capture, laneDetector, outputFileName):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'{outputFileName}.mp4', fourcc, 20.0, (1280,720))

    while capture.isOpened():
        try:
            ret, frame = capture.read()
        except:
            print("Exception at reading capture, continuing")
            continue

        if ret:
            outputImage = laneDetector.detect_lanes(frame)
            out.write(outputImage)
        else:
            break
    
    out.release();
