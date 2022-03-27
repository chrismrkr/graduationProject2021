import numpy as np
import pandas as pd

def data_EDA(X_trains):
    ret = []
    for idx, X_train in enumerate(X_trains):
        tmp = np.copy(X_train).T
        tmp = pd.DataFrame(tmp)

        # middle hip
        tmp[31] = (tmp[16] + tmp[22]) / 2
        tmp[32] = (tmp[17] + tmp[23]) / 2

        # middle knee
        tmp[33] = (tmp[24] + tmp[18]) / 2
        tmp[34] = (tmp[19] + tmp[25]) / 2

        ret.append(np.array(tmp.T, np.float32))

    return np.array(ret, np.float32)


def preprocessMissingDataAndConvert2(dataFrame):
    train = []
    for uid in dataFrame['id'].unique():
        tmp = np.array(dataFrame[dataFrame['id'] == uid].iloc[:, 2:], np.float32).T
        train.append(tmp)
    train = np.array(train, np.float32)

    # 결측치 처리
    for idx, data in enumerate(train):
        for r in range(0, 30):
            t = 0
            while t < 90:
                if data[r][t] == -1.0:
                    left = 0
                    right = 0
                    missing_time_array = []
                    if t - 1 >= 0:
                        left = data[r][t - 1]

                    while (t < 90) and (data[r][t] == -1.0):
                        missing_time_array.append(t)
                        t += 1

                    if t == 90:  # right 없음
                        for time in missing_time_array:
                            data[r][time] = left
                    else:  # right 있음
                        right = data[r][t]
                        if left == 0:
                            for time in missing_time_array:
                                data[r][time] = right
                        else:
                            for time in missing_time_array:
                                data[r][time] = (left + right) / 2;
                t += 1

    train = data_EDA(train)
    train_cpy = np.copy(train)

    result_df = pd.DataFrame(np.transpose(train_cpy, (0, 2, 1)).reshape(-1, 34),
                             index=np.repeat(np.arange(train_cpy.shape[0]), train_cpy.shape[2]))
    result_df = result_df.reset_index()

    return result_df[['index', 2, 3, 10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 28, 29]]
