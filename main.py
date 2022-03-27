import os
import cv2
import numpy as np
from flask import Flask, request
from datetime import datetime
import pickle, json
import pandas as pd
from flask_uploads import UploadSet, configure_uploads
from werkzeug.utils import secure_filename
import flask 

from tensorflow.keras.models import load_model
from movementFrameDetection import detectingMotion
from pushupFrameDetection import capturingPushup
from modifyFrameLength import modifyFrameLength

from extractKeyPoint import makeTrainImagesKeyPoint2DataFrame, makeTrainImagesKeyPoint2DataFrame_multProcess
from preprocessMissingData import preprocessMissingDataAndConvert2
from getPushupInputData import convert2AngleData, convertDataFrame2Numpy
import time
import multiprocessing


''' create flask server'''
app = Flask(__name__)
app.config['UPLOADED_MEDIA_DEST'] = os.path.realpath('.') + '/uploads'

media = UploadSet('media', ('mp4', 'mpeg', 'mov'))
configure_uploads(app, (media, ))

@app.route("/")
def index():
    return "postureCorrectionPage"

@app.route("/savevideo", methods=['POST'])
def saveVideo():
    userId = request.form['UserID']
    print("userID:" , userId)

    video = request.files['file']
    print("video:", video)

    filename = secure_filename(video.filename)  # Secure the filename to prevent some kinds of attack
    media.save(video, name=filename)
    
    response = flask.make_response(flask.send_from_directory(os.path.realpath('.')+"/uploads", filename))
    response.headers['videoHashKey'] = filename
    response.headers['status'] = "True"
    return response
    

@app.route("/savepushup", methods=['POST'])
def savePushup():
    userId = request.form['UserID']
    video = request.files['file']

    filename = secure_filename(video.filename)
    videoFileNameWithExtension = userId + str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + str(datetime.now().hour) + str(datetime.now().minute) \
                + str(datetime.now().second) + filename

    media.save(video, name = videoFileNameWithExtension)

    print(userId)
    print(videoFileNameWithExtension)

    videoFileName = videoFileNameWithExtension.split(".")[0] # 확장자 떼고 비디오 파일명만 저장 -> Key값 사용
    input_URL = os.path.realpath(".") + "/uploads"
    output_URL = os.path.realpath(".") + "/tmp"


    # videoFileName으로 uploads, tmp, runs 디렉토리에 새 디렉토리 생성됨.

    detectingMotion(input_URL, output_URL, videoFileName)  # movementFrame Detection: /tmp/videofileName 디렉토리에 movementframe들이 저장됨
    if len(os.listdir(output_URL+"/"+videoFileName)) == 0:
        # motion이 인식되지 않은 경우 예외처리
        print("no motion founded!")
        os.rmdir(output_URL+"/"+videoFileName)
        response = flask.make_response("fail")
        response.headers['status']="False"
        return response

    sourceFile, orgFileName_list, result = capturingPushup(output_URL + "/" + videoFileName)  # yoloV5 Pushup Detection

    result_URI, resultImage_List = modifyFrameLength(sourceFile, orgFileName_list, result, videoFileName)  # 프레임 길이 조정
    # 결과: motion detection에 의한 90 frame 디렉토리 경로, 90 frame(list)

    # flush motion captured Cash
    print("flush motion captured Cash")
    if os.path.exists(output_URL + "/" + videoFileName):
        tmpFile_list = os.listdir(output_URL +"/" +videoFileName)
        for tmpFile in tmpFile_list:
            os.remove(output_URL + "/"  + videoFileName + "/" + tmpFile)
        os.rmdir(output_URL + "/" + videoFileName)

    # flush yoloV5 Cash
    print("flush yoloV5 Cash")
    if os.path.exists(os.path.realpath('.')+"/runs/"+videoFileName):
        remove_file_list = os.listdir(os.path.realpath('.')+"/runs/" + videoFileName)
        for remove_file in remove_file_list:
            os.remove(os.path.realpath(".")+"/runs/" + videoFileName + "/" + remove_file)
        os.rmdir(os.path.realpath(".") + "/runs/" + videoFileName)

    if len(resultImage_List) == 0:
        # pushup이 인식되지 않은 경우 예외처리
        print("pushup not found")
        response = flask.make_response("fail")
        response.headers['status']="False"
        return response

    out = cv2.VideoWriter(os.path.realpath(".")+"/tmpvideo/"+videoFileName+".mp4", cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))
    for idx, img in enumerate(resultImage_List):
        out.write(img)
    out.release()

    response = flask.make_response(flask.send_from_directory(os.path.realpath(".")+"/tmpvideo/", videoFileName+".mp4"))
    response.headers['videoHashKey'] = videoFileName
    response.headers['status'] = "True"
    return response

@app.route("/correctpushup", methods=['POST'])
def correctPushup():
    content = request.get_json()
    videoHashValue = content['videoHashValue']
    respo = content['keepGoing']
    print(videoHashValue, respo)


    os.remove(os.path.realpath('.')+"/tmpvideo/"+videoHashValue + ".mp4") 

    if respo == "false":
        print("flush images in dest directory")
        destImage_list = os.listdir(os.path.realpath('.')+"/dest/"+"/images/"+videoHashValue)
        
        for filename in destImage_list:
            os.remove(os.path.realpath('.')+"/dest/"+"/images/"+videoHashValue + "/" + filename)
        os.rmdir(os.path.realpath('.')+"/dest/"+"/images/"+videoHashValue)
        
        return "fail"

    inputDirectoryPath = os.path.realpath('.')+"/dest/"+"/images/"+videoHashValue

    # need multiprocessing
    keypoint_df = makeTrainImagesKeyPoint2DataFrame_multProcess(inputDirectoryPath)


    keypointSelected_df = preprocessMissingDataAndConvert2(keypoint_df)
    inputAngleData_df = convert2AngleData(keypointSelected_df)
    inputAngleData_npy = convertDataFrame2Numpy(inputAngleData_df)

    # flush images in dest
    
    print("flush images in dest directory")
    destImage_list = os.listdir(inputDirectoryPath)
    for filename in destImage_list:
        os.remove(inputDirectoryPath+"/"+filename)
    os.rmdir(inputDirectoryPath)
    

    bodyAlignment_pred = UNetPushup_bodyAlignment.predict(inputAngleData_npy[:, :, 0:5])
    bodyAlgnment_result = pd.DataFrame(np.round(bodyAlignment_pred, 3), columns=["body0", "body1", "body2"])
    armPosition_pred = UNetPushup_armPosition.predict(inputAngleData_npy[:, :, 0:5])
    armPosition_result = pd.DataFrame(np.round(armPosition_pred, 3), columns=["arm0", "arm1", "arm2"])
    elbowAngle_pred = UNetPushup_elbowAngle.predict(inputAngleData_npy[:, :, 2:7])
    elbowAngle_result = pd.DataFrame(np.round(elbowAngle_pred, 3), columns=["elbow0", "elbow1"])
    postureCompleteness_pred = UNetPushup_postureCompleteness.predict(inputAngleData_npy[:, :, 0:5])
    postureCompleteness_result = pd.DataFrame(np.round(postureCompleteness_pred, 3), columns=["pc0", "pc1"])
    print(pd.concat([bodyAlgnment_result, elbowAngle_result, armPosition_result, postureCompleteness_result], axis=1))
    print(inputAngleData_npy.shape)
    
    print(bodyAlignment_pred)
    print(armPosition_pred)
    print(elbowAngle_pred)
    print(postureCompleteness_pred)

    return flask.jsonify({
        'bodyAlignment':{
            'good': str(bodyAlignment_pred[0][0]),
            'high': str(bodyAlignment_pred[0][1]),
            'low' : str(bodyAlignment_pred[0][2])
            },
        'armPosition':{
            'good': str(armPosition_pred[0][0]),
            'high': str(armPosition_pred[0][1]),
            'low' : str(armPosition_pred[0][2])
            },
        'elbowAngle': {
            'good': str(elbowAngle_pred[0][0]),
            'bad' : str(elbowAngle_pred[0][1])
            },
        'postureCompleteness': {
            'good': str(postureCompleteness_pred[0][0]),
            'bad' : str(postureCompleteness_pred[0][1])
            }
    })    
    

    
''' run flask server '''
if __name__ == "__main__":
    port = 3054
    
    UNetPushup_bodyAlignment = load_model(os.path.realpath('.')+"/unet"+"/UNetBodyAlignment_211009v1.h5")
    UNetPushup_armPosition = load_model(os.path.realpath('.')+"/unet"+"/UNETArmPosition_211009v1.h5")
    UNetPushup_elbowAngle = load_model(os.path.realpath('.')+"/unet"+"/UNETElbowAngle_211009v1.h5")
    UNetPushup_postureCompleteness = load_model(os.path.realpath('.')+"/unet"+"/UNETPostureCompleteness_211009v1.h5")
    app.run(host='0.0.0.0', port=port)
    

