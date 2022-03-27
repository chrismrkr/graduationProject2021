import cv2, os
import numpy as np


def to_str(idx):
    idx = str(idx)
    if len(idx) == 1:
        idx = '00000' + idx
    elif len(idx) == 2:
        idx = '0000' + idx
    elif len(idx) == 3:
        idx = '000' + idx
    elif len(idx) == 4:
        idx = '00' + idx
    elif len(idx) == 5:
        idx = '0' + idx
    return idx


def detectingMotion(input_path, output_path, fileName):
    threshold = 25
    maxDiff = 250

    pre, current, post = None, None, None

    cap = cv2.VideoCapture(input_path + "/" + fileName + ".mp4")
    cap.get
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if cap.isOpened():
        hasFrame, pre = cap.read()
        hasFrame, current = cap.read()
        os.mkdir(output_path + "/" + fileName)
        idx = 0

        while hasFrame:
            hasFrame, post = cap.read()
            if hasFrame:
                postCopy = post.copy()
            if not hasFrame:
                break

            pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

            # 이전-현재 Frame, 현재-이후 Frame 픽셀 차이 계산
            diff1 = cv2.absdiff(pre_gray, current_gray)
            diff2 = cv2.absdiff(current_gray, post_gray)

            # threshold 값보다 작으면 0으로 변환(프레임에 차이가 없다는 것을 의미함.)
            hasFrame, diff1_t = cv2.threshold(diff1, threshold, 255, cv2.THRESH_BINARY)
            hasFrame, diff2_t = cv2.threshold(diff2, threshold, 255, cv2.THRESH_BINARY)

            # and 비트연산 수행(차이가 있는 픽셀: 1)
            diff = cv2.bitwise_and(diff1_t, diff2_t)

            # 노이즈 처리
            k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

            diffCount = cv2.countNonZero(diff)
            if diffCount > maxDiff:  # 동작 인식된 경우
                notZero = np.nonzero(diff)

                # tmp 디렉토리에 저장
                cv2.imwrite(output_path + "/" + fileName + "/" + fileName + "_" + to_str(idx) + ".jpg", postCopy)
                idx += 1

            pre = current
            current = post
