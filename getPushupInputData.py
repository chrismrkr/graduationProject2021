import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import atan2, degrees
from tqdm import tqdm

def get_angle(x1, y1, cx, cy, x2, y2):
    angle_list = []
    for i in range(0, len(x1)):
        X1 = x1[i]
        Y1 = y1[i]
        CX = cx[i]
        CY = cy[i]
        X2 = x2[i]
        Y2 = y2[i]
        deg1 = (360 + degrees(atan2(X1 - CX, Y1 - CY))) % 360
        deg2 = (360 + degrees(atan2(X2 - CX, Y2 - CY))) % 360
        angle = deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)
        angle_list.append(angle)
    return angle_list

def getTimeDiff(X):
    retList = []
    for k in range(0, len(X) // 90):
        subRet = []
        ex = X[k * 90]
        for i in range(k * 90, (k + 1) * 90):
            subRet.append(X[i] - ex)
            ex = X[i]

        retList = retList + subRet

    return retList


def convert2AngleData(org_dataFrame):
    dest_dataFrame = pd.DataFrame()

    dest_dataFrame['index'] = org_dataFrame['index'].values
    # 가슴-힙-무릎 각도
    dest_dataFrame['chest2hip2kneeAngle'] = get_angle(org_dataFrame[28].values, org_dataFrame[29].values,
                                                      org_dataFrame[22].values, org_dataFrame[23].values,
                                                      org_dataFrame[24].values, org_dataFrame[25].values)
    # 목-가슴-힙 각도
    dest_dataFrame['neck2chest2hipAngle'] = get_angle(org_dataFrame[2].values, org_dataFrame[3].values,
                                                      org_dataFrame[28].values, org_dataFrame[29].values,
                                                      org_dataFrame[22].values, org_dataFrame[23].values)

    # 손목-어깨 x좌표 차이 -> 손목-가슴-힙 각도
    dest_dataFrame['wrist2chest2hipAngle'] = get_angle(org_dataFrame[14].values, org_dataFrame[15].values,
                                                       org_dataFrame[28].values, org_dataFrame[29].values,
                                                       org_dataFrame[22].values, org_dataFrame[23].values)

    # 어깨-팔꿈치-손목 각도
    dest_dataFrame['shoulder2elbow2wristAngle'] = get_angle(org_dataFrame[10].values, org_dataFrame[11].values,
                                                            org_dataFrame[12].values, org_dataFrame[13].values,
                                                            org_dataFrame[14].values, org_dataFrame[15].values)
    # 손목-어깨 y좌표 차이 -> 손목-어깨-가슴 각도
    dest_dataFrame['wrist2shoulder2chestAngle'] = get_angle(org_dataFrame[14].values, org_dataFrame[15].values,
                                                            org_dataFrame[10].values, org_dataFrame[11].values,
                                                            org_dataFrame[28].values, org_dataFrame[29].values)
    # 팔꿈치 X 시간 차이
    dest_dataFrame['elbowXtimeDiff'] = getTimeDiff(org_dataFrame[12].values)

    # 팔꿈치 Y 좌표 차이
    dest_dataFrame['elbowYtimeDiff'] = getTimeDiff(org_dataFrame[13].values)

    return dest_dataFrame


def convertDataFrame2Numpy(org_dataFrame):
    X_train = []
    for uid in org_dataFrame['index'].unique():
        tmp1 = np.array(org_dataFrame[org_dataFrame['index'] == uid].iloc[:, 1:], np.float32).T
        X_train.append(tmp1)
    X_train = np.array(X_train, np.float32)
    X_train = np.transpose(X_train, (0, 2, 1))
    return X_train


def convertNumpy2DataFrame(org_X, org_Y):
    ret_df = pd.DataFrame(columns=['index', 'chest2hip2kneeAngle', 'neck2chest2hipAngle',
                                   'wrist2chest2hipAngle', 'shoulder2elbow2wristAngle', 'wrist2shoulder2chestAngle',
                                   'elbowXtimeDiff', 'elbowYtimeDiff',
                                   'bodyAlignment', 'elbowAngle', 'armPosition', 'postureCompleteness'])
    for i in range(org_X.shape[0]):
        tmp = np.hstack([np.expand_dims(np.array([i] * 90), axis=1), org_X[i]])
        tmpY = []
        for idx in range(0, 90):
            tmpY.append(org_Y[i])
        tmp = np.hstack([tmp, np.array(tmpY)])
        ret_df = pd.concat([ret_df, pd.DataFrame(tmp, columns=['index', 'chest2hip2kneeAngle', 'neck2chest2hipAngle',
                                                               'wrist2chest2hipAngle', 'shoulder2elbow2wristAngle',
                                                               'wrist2shoulder2chestAngle',
                                                               'elbowXtimeDiff', 'elbowYtimeDiff', 'bodyAlignment',
                                                               'elbowAngle', 'armPosition', 'postureCompleteness'])])

    return ret_df