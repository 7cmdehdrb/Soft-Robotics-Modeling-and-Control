아래는 **CC 기구학을 제외하고**, 논문에서 제시한 **Vision 기반 3D Shape Estimation, 즉 ASES(Accurate Shape Estimation System)**만 정리한 내용이다.

## 1. 핵심 개념

이 논문에서 Vision 기반 Shape 추정은 **스테레오 카메라(ZED)를 이용해 소프트 매니퓰레이터의 실제 중심선(backbone)을 3D 좌표로 복원하는 방법**이다. 저자들은 이를 **ASES based on vision**이라고 부른다. 이 방법은 별도의 수학적 기구학 모델 없이, 카메라 이미지에서 매니퓰레이터의 형상을 직접 검출하고 3D로 재구성한다. 

즉, 흐름은 다음과 같다.

**스테레오 이미지 획득 → 매니퓰레이터 윤곽 추출 → SOM으로 중심선 점 추출 → 좌우 영상 대응점 매칭 → 삼각측량으로 3D 좌표 복원 → spline fitting으로 연속 형상 표현**

---

## 2. Vision 시스템 구성

비전 시스템은 비교적 단순하다. 논문에서는 **ZED stereo camera** 하나를 삼각대에 고정하고, 카메라 시야 안에 소프트 매니퓰레이터의 전체 작업 공간이 들어오도록 배치한다. 워크스테이션에서 영상 처리와 3D 복원을 수행한다. 

중요한 점은 이 Vision 기반 결과가 논문 전체 프레임워크에서는 **고정밀 기준값 또는 학습용 출력 데이터** 역할을 한다는 것이다. 즉, 실시간 최종 시스템에서는 DNN이 이를 모방하도록 학습되지만, Vision 기반 ASES 자체는 정확한 3D shape를 얻기 위한 기준 측정 시스템으로 사용된다.

---

## 3. 단계 1: 이미지 전처리 및 윤곽 추출

먼저 카메라에서 좌우 이미지를 취득한다. 논문에서는 ZED-API가 제공하는 factory parameter를 이용해 카메라 보정을 수행하고, **1280 × 720 해상도의 왜곡 보정 이미지를 사용**한다. 이후 매니퓰레이터를 배경에서 분리하기 위해 영상 전처리를 수행한다. 

전처리 과정은 다음 순서다.

1. **Gaussian filter**로 영상 노이즈 제거
2. RGB 이미지를 **HSV 색공간**으로 변환
3. 매니퓰레이터 색상을 threshold로 설정하여 **mask 처리**
4. binary image 생성
5. **morphological filter**로 고립 픽셀 제거 및 경계 smoothing
6. **Canny edge detection**으로 매니퓰레이터의 외곽 contour 추출

여기서 중요한 것은 저자들이 매니퓰레이터의 형상 전체를 직접 추적하는 것이 아니라, 먼저 **외곽 contour point set**을 만든다는 점이다. Fig. 9에서는 원본 이미지 → threshold 결과 → contour extraction 결과가 순서대로 제시되어 있다. 

---

## 4. 단계 2: SOM을 이용한 중심선 추출

윤곽선만으로는 3D shape를 표현하기 어렵다. 소프트 매니퓰레이터의 실제 형상은 외곽선이 아니라 **중심선(backbone 또는 centerline)**으로 표현하는 것이 적절하다. 논문에서는 매니퓰레이터가 균일하고 대칭적인 관형 구조라는 점을 이용해, contour로부터 중심선을 추정한다. 

이를 위해 사용한 알고리즘이 **SOM(Self-Organizing Map)**이다.

SOM의 역할은 간단히 말하면 다음과 같다.

> contour point cloud를 입력으로 받아, 그 가운데를 따라가는 대표 점들을 자동으로 배치하는 클러스터링 알고리즘

논문에서는 SOM의 입력을 2D contour 좌표로 둔다. 즉, 입력 샘플은 이미지 평면상의 contour point이고, 출력 뉴런은 중심선 위의 대표 점이다. 저자들은 최종적으로 **7개의 중심점**을 사용한다. Table 1에 따르면 SOM 파라미터는 neighborhood radius (S=3), 초기 학습률 (\alpha(0)=0.01), training number (\lambda=15), 중심점 개수 (b=7)이다. 

SOM이 중요한 이유는 두 가지다.

첫째, contour에서 중심선을 직접 계산하는 단순 기하 방법보다 유연하다. 소프트 매니퓰레이터가 크게 휘거나 곡률이 일정하지 않아도, contour 분포를 따라 중심점들을 adaptive하게 배치할 수 있다.

둘째, SOM의 ordering property 때문에 좌우 스테레오 이미지에서 추출된 중심점들이 순서대로 대응된다. 즉, 왼쪽 이미지의 (W_k^L)와 오른쪽 이미지의 (W_k^R)를 별도의 복잡한 sorting algorithm 없이 대응점으로 사용할 수 있다. 

---

## 5. 단계 3: 스테레오 삼각측량 기반 3D 복원

좌우 이미지에서 중심점 대응쌍을 얻으면, 각 중심점의 3D 좌표를 계산할 수 있다. 논문에서는 ZED 카메라의 좌우 카메라가 평행하게 배치되어 있다는 점을 이용해 **stereo triangulation**을 수행한다. 

각 backbone point를 다음과 같이 둔다.

[
Q_k = [Q_k^x, Q_k^y, Q_k^z]^T
]

좌우 이미지에서 해당 점의 x좌표를 각각 (x_k^L), (x_k^R)라고 하면 disparity는 다음과 같다.

[
\tau_k = x_k^L - x_k^R
]

깊이값은 다음 식으로 계산된다.

[
Q_k^z = \frac{fT_x}{\tau_k}
]

여기서 (f)는 초점거리, (T_x)는 좌우 카메라 사이의 baseline이다. 이후 x, y 좌표는 다음처럼 복원된다.

[
Q_k^x = \frac{x_k^L Q_k^z}{f}
]

[
Q_k^y = \frac{y_k^L Q_k^z}{f}
]

이 과정을 7개의 중심점에 대해 반복하면, 최종적으로 backbone을 나타내는 3D point set이 생성된다.

[
Q = [Q_1, Q_2, ..., Q_b] \in \mathbb{R}^{3b}
]

논문에서는 (b=7)이므로, 최종 출력은 **7개 점 × 3차원 = 21차원 shape vector**로 볼 수 있다. 

---

## 6. 좌표계 변환

삼각측량으로 얻은 3D 좌표는 기본적으로 **왼쪽 카메라 좌표계** 기준이다. 하지만 이후 shape estimation 결과를 로봇 또는 시스템 좌표계에서 사용하려면 좌표계를 맞춰야 한다.

이를 위해 논문에서는 Ordinary Least Squares를 이용해, 카메라 좌표계에서 global coordinate로 가는 homogeneous transformation matrix (T_{L-g})를 구한다. 이 변환을 적용해 최종 ASES 결과 (\hat{Q})를 얻는다. 

정리하면,

[
\text{camera coordinate에서 얻은 } Q
]

를

[
\text{global coordinate의 } \hat{Q}
]

로 변환하는 과정이 포함된다.

---

## 7. 연속 형상 표현

3D로 복원된 것은 기본적으로 7개의 discrete backbone point이다. 그러나 소프트 매니퓰레이터는 연속체이므로, 점들만으로는 형상이 끊겨 보인다.

그래서 논문에서는 **improved cubic spline fitting**을 적용하여 7개 중심점을 통과하는 연속 곡선 형태로 soft manipulator의 3D shape를 표현한다. Fig. 12(c)에서 각 pose별로 복원된 3D 중심선이 표시되어 있으며, 저자들은 실험 결과와 시뮬레이션 결과가 잘 일치한다고 설명한다. 

---

## 8. Vision 기반 방법의 장점

이 방법의 장점은 명확하다.

첫째, **기구학 모델이 필요 없다.**
매니퓰레이터의 길이, 곡률, 굽힘각 등을 수식으로 모델링하지 않고, 이미지에서 직접 형상을 추정한다.

둘째, **비접촉식 측정이다.**
센서를 매니퓰레이터 내부에 삽입하지 않아도 되므로, 소프트 로봇의 유연성을 해치지 않는다.

셋째, **큰 변형과 variable curvature 형상에도 대응 가능하다.**
CC assumption처럼 일정 곡률을 가정하지 않기 때문에, 실제 매니퓰레이터가 비정상적인 곡률이나 3D torsion을 보이더라도 시각적으로 관측 가능한 범위에서는 형상 복원이 가능하다.

넷째, 논문에서는 ASES 결과가 실제 실험 형상과 잘 일치한다고 보고한다. 특히 Fig. 12에서는 4개의 대표 pose에 대해 contour extraction, SOM centerline extraction, 3D reconstruction 결과를 제시한다. 

---

## 9. Vision 기반 방법의 한계

하지만 이 방법은 실제 적용성 측면에서 제한이 크다.

가장 큰 문제는 **환경 의존성**이다. 조명 조건, 배경 복잡도, 매니퓰레이터와 배경의 색상 대비가 contour extraction 성능에 직접 영향을 준다. 논문에서도 실험을 위해 흰색 단일 배경과 보조 조명을 사용했다고 설명한다. 조명이 나쁜 경우에는 contour 자체가 잘못 추출되고, 그 결과 SOM 중심점 추출도 실패한다. 

두 번째 문제는 **self-occlusion**이다. 매니퓰레이터가 뒤쪽으로 휘거나 자기 자신을 가리는 자세가 되면 카메라에서 전체 contour를 올바르게 볼 수 없다. 이 경우 중심선 추출과 3D 복원이 실패한다. Fig. 13에서는 poor light condition과 self-occlusion 상황에서 ASES가 제대로 동작하지 않는 사례가 제시된다. 

따라서 이 논문의 결론은 다음에 가깝다.

> Vision 기반 ASES는 정확도는 높지만, 환경 조건이 통제된 실험실 환경에 적합하다. 실제 실시간 운용 또는 제약 환경에서는 단독 사용이 어렵다.

---

## 10. 전체 요약

이 논문의 Vision 기반 Shape 추정은 **스테레오 카메라 기반 3D backbone reconstruction 방법**이다. 먼저 ZED 카메라로 좌우 이미지를 획득하고, HSV thresholding, morphological filtering, Canny edge detection을 통해 매니퓰레이터 contour를 추출한다. 이후 SOM 알고리즘으로 contour 내부의 중심선 대표점 7개를 계산하고, 좌우 이미지의 대응점 disparity를 이용해 삼각측량으로 각 중심점의 3D 좌표를 복원한다. 마지막으로 좌표계 변환과 spline fitting을 거쳐 연속적인 3D shape를 표현한다.

기술적으로는 **image processing + SOM centerline extraction + stereo triangulation**의 조합이다. 이 방법은 수학적 기구학 모델에 의존하지 않고 실제 형상을 직접 관측하므로, variable curvature나 큰 변형 상황에서 정확한 기준 형상을 제공할 수 있다. 다만 조명, 배경, occlusion에 민감하기 때문에, 논문에서는 이를 실시간 최종 센서로 사용하기보다는 DNN 학습을 위한 고정밀 기준 데이터로 활용한다.
