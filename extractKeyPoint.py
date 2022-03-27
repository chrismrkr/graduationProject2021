import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import time
from multiprocessing import Pool
protoFile = os.path.realpath('.') + "/OpenPose/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = os.path.realpath('.') + "/OpenPose/pose/mpi/pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
threshold = 0.1
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

feature_name = ['head_', 'neck_',  'Rshoulder_', 'Relbow_', 'Rwrist_', 'Lshoulder_', 'Lelbow_', 'Lwrist_', 'Rhip_','Rknee_', 'Rankle_', 'Lhip_', 'Lknee_', 'Lankle_', 'chest_']


inWidth = 368
inHeight = 368

def extractPoint(frameTuple):
    frame = frameTuple[0]
    time = frameTuple[1]

    ret = {'id': 1}
    ret['time'] = time
    
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # inwidth : 1920, inheight = 1080
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    
    for i in range(nPoints):
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold :
            ret[feature_name[i]+'x'] = x
            ret[feature_name[i]+'y'] = y
            # Add the point to the list if the probability is greater than the threshold
        else :
            ret[feature_name[i]+'x'] = -1.0
            ret[feature_name[i]+'y'] = -1.0
    return ret

def makeTrainImagesKeyPoint2DataFrame_multProcess(data_path):
    data_list = os.listdir(data_path)
    result_dic = {
    'id': [], 'time': [],
    'head_x': [], 'head_y': [], 'neck_x': [], 'neck_y': [], 'Rshoulder_x': [],
       'Rshoulder_y': [], 'Relbow_x': [], 'Relbow_y': [], 'Rwrist_x': [], 'Rwrist_y': [],
       'Lshoulder_x': [], 'Lshoulder_y': [], 'Lelbow_x': [], 'Lelbow_y': [], 'Lwrist_x': [],
       'Lwrist_y': [], 'Rhip_x': [], 'Rhip_y': [], 'Rknee_x': [], 'Rknee_y': [], 'Rankle_x': [],
       'Rankle_y': [], 'Lhip_x': [], 'Lhip_y': [], 'Lknee_x': [], 'Lknee_y': [], 'Lankle_x': [],
       'Lankle_y': [], 'chest_x': [], 'chest_y': [],

    }

    ID = 1
    time = 0

    frameList = []
    for idx, image_file in enumerate(data_list):
        frameList.append((cv2.imread(data_path+"/"+image_file), idx))

    pooler = Pool(processes=24)
    multiprocessedResult = tqdm(pooler.imap(extractPoint, frameList))
    
    print(multiprocessedResult)

    
    for ret in multiprocessedResult:
        result_dic['id'].append(ret.get('id'))
        result_dic['time'].append(ret.get('time'))
        for i in range(nPoints):
            result_dic[feature_name[i]+'x'].append(ret.get(feature_name[i]+'x'))
            result_dic[feature_name[i]+'y'].append(ret.get(feature_name[i]+'y'))
    print(pd.DataFrame(result_dic))
    return pd.DataFrame(result_dic)
    



    

def makeTrainImagesKeyPoint2DataFrame(data_path):
    # train_data_path : ' ~/dest/images/"filename" '
    data_list = os.listdir(data_path)
    result_dic = {
    'id': [], 'time': [],
    'head_x': [], 'head_y': [], 'neck_x': [], 'neck_y': [], 'Rshoulder_x': [],
       'Rshoulder_y': [], 'Relbow_x': [], 'Relbow_y': [], 'Rwrist_x': [], 'Rwrist_y': [],
       'Lshoulder_x': [], 'Lshoulder_y': [], 'Lelbow_x': [], 'Lelbow_y': [], 'Lwrist_x': [],
       'Lwrist_y': [], 'Rhip_x': [], 'Rhip_y': [], 'Rknee_x': [], 'Rknee_y': [], 'Rankle_x': [],
       'Rankle_y': [], 'Lhip_x': [], 'Lhip_y': [], 'Lknee_x': [], 'Lknee_y': [], 'Lankle_x': [],
       'Lankle_y': [], 'chest_x': [], 'chest_y': [],

    }

    ID = 1
    time = 0

    frameList = [] 
    for idx, image_file in enumerate(data_list):
        frameList.append((cv2.imread(data_path+"/"+image_file), time))


# ------------------------------------------------------------------------------
    for image_file in tqdm(data_list):
        frame = cv2.imread(data_path+"/"+image_file)
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        # inwidth : 1920, inheight = 1080
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        result_dic['id'].append(ID)
        result_dic['time'].append(time)
        for i in range(nPoints):
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold :
                result_dic[feature_name[i]+'x'].append(x)
                result_dic[feature_name[i]+'y'].append(y)
                # Add the point to the list if the probability is greater than the threshold
            else :
                result_dic[feature_name[i]+'x'].append(-1.0)
                result_dic[feature_name[i]+'y'].append(-1.0)
        time += 1
    return pd.DataFrame(result_dic)

