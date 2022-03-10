### Hongik Univ, Computer Engineering Graduation Project 2021
### 딥러닝을 활용한 헬스 자세 교정 프로그램

***
#### 프로젝트 성과

2021 홍익대학교 컴퓨터공학과 졸업프로젝트 전시회 최우수상 
***

+ **개발 배경 및 서론**

유튜브와 같은 소셜미디어가 대중화됨에 따라 사람들이 쉽게 홈 트레이닝 및 헬스에 대한 정보를 쉽게 얻을 수 있게 되었습니다.

그러나, 매체를 통해 헬스에 대한 정보를 얻었다 하더라도 적절한 피드백이 없다면 본인이 헬스를 제대로 하고 있는지 정확히는 알 수는 없습니다.

이에 헬스 자세를 인식하고 피드백을 해주는 프로그램의 필요성을 느껴 이 프로젝트를 시작했습니다.

물론, 수 많은 헬스 자세에 대한 피드백을 모두 제공하는 종합 프로그램을 만드는 것은 학습 데이터 수집의 어려움 때문에 결국 한계가 있었습니다.

그럼에도, 여러 헬스 자세 중 개발이 완료된 푸시업 자세 교정 프로그램을 집중해 본 프로젝트의 내용과 결과에 대해 설명하도록 합니다.

***

#### 프로그램 아키텍처

프로그램은 프레임 감지 기능, 키 포인트 추출, 그리고 자세평가 세 단계로 이루어집니다.

사용자가 본인이 운동한 푸시업한 영상을 프로그램에 업로드하면 세 단계를 거친 후, 피드백 결과를 받을 수 있습니다.

개괄적인 코드는 아래와 같습니다.

```python
  # inputFilePath에 피드백 받고자 하는 운동 자세가 담긴 영상 파일이 있다.

  detectPushUp(inputFilePath, tmpFilePath) // 동영상에서 운동 동작(푸시업)이 나타난 부분(프레임)만 추출한다.
  timeSeriesDataFrame = getTimeSeriesKeyPoint(tmpFilePath) // 추출된 프레임으로 부터 인체 키 포인트 좌표를 추출한다.
  getResult(timeSeriesDataFrame, outputFilePath) // 추출된 시계열의 키 포인트 좌표들을 조작힌 후, 딥러닝을 통해 피드백 결과를 저장한다. 
```

+ **프레임 감지 기능(detectPushup)**

프레임 감지 기능의 결과는 아래의 두 gif a와 b를 통해 확인할 수 있습니다.
***
<img width="100%" src="https://user-images.githubusercontent.com/62477958/157073502-dc301e39-7ac3-4499-b13b-6e96bb707266.gif"/>

**a. 촬영된 동영상**


<img width="100%" src="https://user-images.githubusercontent.com/62477958/157073574-938bf746-a1dc-43ea-8ab3-eb58e5d6abe1.gif"/>

**b. 프레임 감지 기능이 적용된 촬영된 동영상**
***
만약, 프로그램 사용자가 본인의 푸시업 자세에 대해 피드백을 받기 위해 동영상을 촬영했다고 가정합니다.

촬영된 동영상(gif a)에는 피드백이 필요한 푸시업 동작 뿐만 아니라 불필요한 부분(ex. 걸어가는 부분, 푸시업 자세를 잡기 위한 준비동작 등)이 포함됩니다.

그러므로, 촬영된 영상으로부터 불필요한 부분을 제거해야 합니다. 이러한 제거 과정을 프레임 감지 기능이 담당합니다.

프레임 감지 기능은 컴퓨터 비젼 라이브러리 OpenCV와 객체 탐지 딥러닝 모델인 YOLOv5를 사용해 구현했습니다.

(https://github.com/ultralytics/yolov5)

동영상 내 움직임이 나타나지 않은 프레임 구간을 판정해 제거하기 위해 OpenCV 라이브러리를 활용했고,

푸시업이 나타난 부분을 탐지하기 위해 YOLOv5 모델을 학습시켜 사용했습니다.


+ **키 포인트 추출 기능(getTimeSeriesKeyPoint)**

추출된 동영상(프레임들)으로부터 인체 키 포인트를 추출합니다. 키 포인트 추출은 OpenPose 라이브러리를 활용했습니다.

(https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/)

추출된 인체 키 포인트 좌표들(ex. 머리, 어깨, 가슴 등)을 이용해 각도 등의 새로운 특성을 생성합니다.

각 프레임마다 인체 키 포인트 좌표를 추출해 새로운 특성을 생성되므로 (프레임 index * 생성된 특성) 형태의 2차원 형태의 배열로 변환됩니다.

새롭게 생성된 배열은 딥러닝을 통해 운동 자세를 평가하기 위해 사용됩니다. 


+ **자세 평가 기능(getResult)**

마지막 단계에서는 변환된 배열을 딥러닝을 통해 평가합니다. 

다양한 모델은 실험한 후, 결과적으로 딥러닝 모델은 U-Net 모델을 채택했습니다.

U-Net 모델은 시계열 특성을 띄는 수면파로부터 램수면 파동과 같은 수면 패턴을 검출하기 위해 사용되었다는 선행연구가 존재합니다.

인체 키 포인트로 부터 변환된 시계열 배열은 여러 정보를 담을 수 있습니다.

가령, 어떤 배열은 정상적인 푸시업 자세라는 특성을 갖고 있을 수 있고, 또 다른 배열은 허리의 위치나 팔꿈치의 각도가 잘못되었다는 정보를 담을 수 있습니다.

U-Net 모델을 통해 학습 및 검증한 결과, 자세에 대한 평가와 피드백은 약 88%의 정확도를 보였습니다.

***

아래의의 두 gif를 통해 자세 평가 기능의 결과를 확인할 수 있습니다.

<img width="90%" src="https://user-images.githubusercontent.com/62477958/157093925-598c5b0d-14c3-4eeb-9f5e-b8d9c6db71f3.gif"/>

**a. 앞서 촬영된 동영상**

<img width="90%" src="https://user-images.githubusercontent.com/62477958/157093982-2b769ca6-8ca5-497b-b1c6-faf72096c093.gif"/>

**b. 피드백에 따라 교정 후 촬영한 동영상**

a는 앞서 프레임 감지 기능을 통해 추출된 것에 자세 교정 기능을 적용한 결과입니다.

엉덩이 위치와 팔꿈치 각도가 잘못되었고, 좀 더 내려가야 한다는 피드백 결과를 받았습니다.

이에 피드백 결과를 반영하여 동영상을 다시 촬영 후 프로그램을 적용한 결과는 b에서 확인할 수 있습니다.

자세 교정을 통해 a와는 달리 적절한 자세라고 판정받았고 gif를 통해서도 a와 b 사이에 푸시업 동작의 차이점을 볼 수 있습니다.

***

#### 프로그램 사용법

```git
  git pull 

```
