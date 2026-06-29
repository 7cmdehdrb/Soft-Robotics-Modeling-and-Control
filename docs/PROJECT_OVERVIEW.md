# 프로젝트 전체 설명

## 1. 프로젝트 목적

이 레포지터리는 소프트 로봇의 **모델링, 센싱, 상태 추정, 제어, 시각화**를 실험하기 위한 연구용 작업 공간이다. 현재 목표는 하나의 완성된 애플리케이션을 제공하는 것이라기보다, 여러 센서와 모델링 방법을 조합해 소프트 로봇의 상태를 추정하고 제어하는 방법을 탐색하는 데 있다.

초기 README의 핵심 목표는 다음과 같이 정리할 수 있다.

1. 소프트 로봇에 물리적으로 안전한 범위의 랜덤 actuation input을 인가한다.
2. 입력 데이터와 출력 데이터를 시간 기준으로 로깅한다.
3. 수집한 데이터를 이용해 state estimation model을 학습한다.
4. 초기 모델로 LSTM 기반 시계열 회귀 모델을 사용한다.
5. 학습된 모델과 센서/비전 시스템을 이용해 로봇 상태를 예측하고 평가한다.

## 2. 현재 레포지터리의 성격

이 레포지터리는 아직 **리서치 단계**이다. 따라서 패키지들이 하나의 안정적인 제품 구조로 유기적으로 결합되어 있다고 보기 어렵다. 각 폴더는 다음과 같은 성격을 가진다.

- 특정 센서 또는 장비를 ROS2에 연결하기 위한 패키지
- 공압 제어, Arduino 통신, 데이터 수집을 위한 실험 코드
- LSTM 기반 상태 추정/마커 위치 예측 모델 학습 코드
- 비전 기반 centerline/shape estimation 실험 코드
- Cosserat rod 기반 해석 모델 및 GUI 실험 코드
- Unity 또는 시뮬레이터 연동을 위한 외부 패키지

따라서 이 문서는 모든 패키지가 이미 동일한 런타임 파이프라인에 들어가 있다고 설명하지 않는다. 대신 각 구성요소가 어떤 연구 질문을 다루는지, 어떤 데이터를 주고받는지, 앞으로 어떻게 연결될 수 있는지를 중심으로 정리한다.

## 3. 연구 파이프라인 개요

프로젝트가 지향하는 큰 흐름은 다음과 같다.

```text
Actuation command 생성
  → Arduino/공압 시스템으로 로봇 구동
  → 센서, OptiTrack, 카메라, depth point cloud에서 상태 관측
  → 데이터 로깅 및 시간 동기화
  → LSTM 또는 다른 모델로 상태/마커 위치/형상 추정
  → ROS2 토픽과 RViz로 결과 시각화
  → 제어 알고리즘 또는 시뮬레이션과 연결
```

현재는 이 흐름의 각 블록이 독립적으로 개발되고 있으며, 일부는 ROS2 패키지로, 일부는 단일 Python 스크립트 또는 Arduino sketch로 존재한다.

## 4. 주요 서브시스템

### 4.1 공압 제어와 Arduino 통신

`src/arduino`는 공압 밸브 제어를 위한 Arduino firmware와 Python serial sender를 포함한다. Arduino는 고정 길이 binary packet을 받아 솔레노이드 밸브와 비례 밸브를 제어하고, Python 스크립트는 랜덤 또는 지정된 제어 값을 전송할 수 있다.

자세한 내용은 [`../src/arduino/README.md`](../src/arduino/README.md)를 참고한다.

### 4.2 LSTM 기반 상태/마커 예측

`src/lstm`은 공압 제어 신호 또는 센서 시계열을 입력으로 받아 소프트 로봇의 마커 위치를 예측하는 LSTM 학습 코드를 포함한다. 기본 아이디어는 과거 window의 입력을 사용해 미래 또는 현재의 marker position을 회귀하는 것이다.

자세한 학습 옵션과 실행법은 [`../src/lstm/README.md`](../src/lstm/README.md)를 참고한다.

### 4.3 ROS2 predictor와 soft finger 제어

`src/psf_control`은 ROS2 패키지 형태로 구성되어 있으며, Arduino 또는 센서 데이터 토픽을 구독해 LSTM 모델의 marker prediction 결과를 `MarkerArray`로 발행하는 구성을 가진다.

자세한 노드 설명과 실행법은 [`../src/psf_control/README.md`](../src/psf_control/README.md)를 참고한다.

### 4.4 OptiTrack/NatNet 연동

`src/natnet`은 OptiTrack Motive에서 송신되는 NatNet 데이터를 ROS2 토픽으로 변환한다. Rigid body와 unlabeled marker를 `MarkerArray`로 발행해 RViz에서 관찰하거나 학습 데이터의 기준값으로 사용할 수 있다.

자세한 통신 방식과 토픽 설명은 [`../src/natnet/README.md`](../src/natnet/README.md)를 참고한다.

### 4.5 Vision 기반 centerline/shape estimation

`src/vision_estimation`은 이미지 기반으로 소프트 로봇의 contour와 centerline을 추정하는 코드를 포함한다. `ros_white_centerline_client.py`는 ROS2 image와 organized `PointCloud2`를 이용해 2D centerline point를 3D 좌표로 변환하고, RViz용 marker와 `PoseArray`를 발행한다.

관련 문서는 다음을 참고한다.

- [`../src/vision_estimation/VISION.md`](../src/vision_estimation/VISION.md): Vision 기반 ASES 개념 정리
- [`../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md`](../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md): ROS2 white centerline client 설명

### 4.6 Cosserat rod 기반 모델링 실험

`src/CR`은 Cosserat rod 기반 소프트 로봇 모델링과 GUI 실험 스크립트를 포함한다. 현재는 물리 파라미터와 기하 파라미터를 별도 Markdown 메모로 관리한다.

관련 문서는 다음을 참고한다.

- [`../src/CR/GEOMETRIC_PARAM.md`](../src/CR/GEOMETRIC_PARAM.md)
- [`../src/CR/PHYSICS_PARAM.md`](../src/CR/PHYSICS_PARAM.md)

### 4.7 Genesis/시뮬레이션 실험

`src/genesis`는 Genesis 기반 예제, ROS 연동 예제, camera pose/joint state GUI publisher 등을 포함한다. 현재 별도 Markdown 문서는 없으며, 실험 스크립트 중심의 폴더로 볼 수 있다.

### 4.8 Unity ROS-TCP Endpoint

`src/ROS-TCP-Endpoint-ROS2v0.7.0`은 Unity와 ROS2를 연결하기 위한 ROS-TCP Endpoint 패키지이다. 외부 패키지 성격이 강하므로 해당 패키지의 원본 문서를 유지한다.

자세한 내용은 [`../src/ROS-TCP-Endpoint-ROS2v0.7.0/README.md`](../src/ROS-TCP-Endpoint-ROS2v0.7.0/README.md)를 참고한다.

## 5. 데이터와 좌표계 관점

이 레포지터리의 실험은 여러 좌표계와 데이터 소스를 함께 다룬다.

| 데이터/좌표계 | 관련 폴더 | 용도 |
| --- | --- | --- |
| 공압 제어 입력 | `src/arduino`, `src/psf_control` | 로봇 actuation command |
| 압력/센서 시계열 | `src/arduino`, `src/lstm`, `src/psf_control` | 학습 입력 또는 상태 관측 |
| OptiTrack marker | `src/natnet`, `src/lstm` | marker position ground truth 또는 시각화 |
| 카메라 RGB/depth | `src/vision_estimation` | centerline/shape estimation |
| RViz marker | `src/natnet`, `src/psf_control`, `src/vision_estimation` | 관측/예측 결과 시각화 |
| TF transform | `docs/tf.md` | 장비 간 좌표계 연결 메모 |

좌표계와 장비 배치가 실험마다 달라질 수 있으므로, TF 명령과 calibration 값은 고정된 제품 설정이라기보다 실험 메모로 취급해야 한다.

## 6. 실행 명령 문서

RealSense point cloud 설정, ROS2 parameter 확인, TF static transform publisher 명령은 [`COMMANDS.md`](COMMANDS.md)와 [`tf.md`](tf.md)에 정리되어 있다.

## 7. 앞으로 문서화가 필요한 부분

현재 레포지터리를 더 안정적으로 사용하려면 다음 문서가 추가되면 좋다.

- 전체 ROS2 launch 구성도
- 데이터 로깅 포맷과 timestamp synchronization 규칙
- 실험별 좌표계 정의와 calibration 절차
- 학습 데이터셋 생성 절차
- LSTM 모델 파일 저장 위치와 predictor 설정 방식
- hardware setup 사진 또는 wiring diagram
