import cv2, os
import numpy as np

def underSize(orgImage_list):
    tmp_list = orgImage_list.copy()
    while len(tmp_list) >= 180:
        tmp2_list = []
        for i in range(0, len(tmp_list), 2):
            tmp2_list.append(tmp_list[i])
        tmp_list = tmp2_list.copy()

    diff = len(tmp_list) - 90
    left, right = diff // 2, diff // 2
    if diff % 2 == 1: left += 1
    left *= 2
    right *= 2

    middle_list = []
    if right == 0:
        middle_list = tmp_list[left:]
    else:
        middle_list = tmp_list[left:-right]

    left_list = []
    for i in range(0, left, 2):
        left_list.append(tmp_list[i])

    right_list = []
    for i in range(len(tmp_list) - right, len(tmp_list), 2):
        right_list.append(tmp_list[i])

    result = left_list + middle_list + right_list

    assert len(result) == 90, "동영상 길이 변환 결과가 올바르지 않습니다."
    return result


def overSize(orgImage_list):
    tmp_list = orgImage_list.copy()
    while len(tmp_list) <= 45:
        print("check")
        print(len(tmp_list))
        tmp2_list = []
        for i in range(0, len(tmp_list)):
            tmp2_list.append(tmp_list[i])
            tmp2_list.append(tmp_list[i])
        tmp_list = tmp2_list.copy()

    diff = 90 - len(tmp_list)

    left = len(tmp_list) // 2
    right = len(tmp_list) // 2
    while diff > 0:
        left -= 1
        diff -= 1
        if diff == 0:
            break
        right += 1
        diff -= 1
    left_list = tmp_list[:left]
    right_list = tmp_list[right:]

    middle_list = []
    for i in range(left, right):
        middle_list.append(tmp_list[i])
        middle_list.append(tmp_list[i])

    result = left_list + middle_list + right_list
    assert len(result) == 90, "동영상 길이 변환 결과가 올바르지 않습니다."
    return result


def toString(idx):
    idx = str(idx)
    if len(idx) == 1: idx = '0' + idx
    return idx


def modifyFrameLength(sourceFile, orgFileName_list, captured_list, dirName):
    start, end = -1, -1
    frameCnt = 0
    noPushupCount = 0

    # 학습 데이터 생성이 아닌 실제 엔진에서는 조작 필요함
    for idx, frame in enumerate(captured_list):
        if (len(frame) >= 1) and (frame[0][5] == 1):
            noPushupCount += 1
            if (noPushupCount >= 15) and (start != -1):
                end = idx - 17
                break
            continue

        if (len(frame) >= 1) and (frame[0][5] == 0):  # frame에서 pushUp이 탐지된 경우
            noPushupCount = 0
            frameCnt += 1
            if (start != -1) and (frameCnt >= 15):  # write end
                end = idx - 3
        else:
            noPushupCount = 0
            frameCnt = 0

        if (start == -1) and (frameCnt == 10):
            # write start
            start = idx - 7

    if start == -1: return "fail", []

    if end == -1: end = len(captured_list) - 1

    destImage_list = []
    for idx in range(start, end + 1):
        destImage_list.append(cv2.imread(sourceFile + "/" + orgFileName_list[idx]))

    if len(destImage_list) > 90:  # 선별된 프레임 수가 90개 초과 -> 압축 필요
        result_list = underSize(destImage_list)

    elif len(destImage_list) < 90:  # 선별된 프레임 수가 90개 미만 -> 확장 필요
        result_list = overSize(destImage_list)

    else:
        result_list = destImage_list.copy()

    result_image_URI = os.path.realpath('.') + "/dest/images/" + dirName + "/"

    os.mkdir(result_image_URI)
    for idx, img in enumerate(result_list):
        if (img.shape[0] != 1080):
            img = cv2.resize(img, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.imwrite(result_image_URI + "processed" + toString(idx) + ".jpg", img)

    return result_image_URI, destImage_list