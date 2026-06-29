# Soft Robotics Modeling and Control 문서 허브

이 폴더는 레포지터리 전체를 이해하기 위한 상위 문서들을 모아 둔 공간이다. 이 레포지터리는 아직 리서치 단계이며, 각 패키지와 실험 코드가 하나의 완성된 제품처럼 강하게 결합되어 있지는 않다. 따라서 문서는 **전체 방향성**, **디렉토리 구조**, **실험 실행 메모**, **패키지별 상세 문서 링크**를 분리해서 정리한다.

## 문서 목록

| 문서 | 목적 |
| --- | --- |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | 프로젝트 전체 목표, 연구 단계에서의 구성, 주요 서브시스템 설명 |
| [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) | 레포지터리 디렉토리와 주요 파일의 역할 설명 |
| [COMMANDS.md](COMMANDS.md) | 자주 사용하는 ROS2 실행/설정 명령 모음 |
| [tf.md](tf.md) | TF static transform publisher 명령 메모 |

## 패키지별 상세 문서

특정 패키지 아래에 있는 Markdown 문서는 해당 패키지의 동작과 사용법을 이해하기 위한 문서이므로 위치를 이동하지 않는다. 대신 이 허브 문서에서 참조한다.

| 위치 | 설명 |
| --- | --- |
| [`../src/arduino/README.md`](../src/arduino/README.md) | Arduino 기반 공압 제어 펌웨어와 시리얼 송신 스크립트 |
| [`../src/lstm/README.md`](../src/lstm/README.md) | 공압/센서 시계열을 이용한 LSTM 마커 위치 예측 모델 학습 |
| [`../src/psf_control/README.md`](../src/psf_control/README.md) | ROS2 기반 soft finger 제어 및 LSTM predictor 노드 |
| [`../src/natnet/README.md`](../src/natnet/README.md) | OptiTrack NatNet 데이터를 ROS2 토픽으로 발행하는 패키지 |
| [`../src/vision_estimation/VISION.md`](../src/vision_estimation/VISION.md) | Vision 기반 3D shape estimation/ASES 개념 정리 |
| [`../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md`](../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md) | `ros_white_centerline_client.py` ROS2 노드 설명 |
| [`../src/CR/GEOMETRIC_PARAM.md`](../src/CR/GEOMETRIC_PARAM.md) | Cosserat rod 모델링에 사용하는 단면 기하 파라미터 메모 |
| [`../src/CR/PHYSICS_PARAM.md`](../src/CR/PHYSICS_PARAM.md) | 재료 물성치 메모 |
| [`../src/ROS-TCP-Endpoint-ROS2v0.7.0/README.md`](../src/ROS-TCP-Endpoint-ROS2v0.7.0/README.md) | Unity ROS-TCP Endpoint 패키지 원본 문서 |

## 문서 관리 원칙

- 레포지터리 전체 설명은 `/docs` 아래에 둔다.
- 패키지 내부의 세부 실행법, API, 실험 조건은 해당 패키지 폴더의 Markdown에 둔다.
- 패키지 내부 Markdown은 이동하지 않고 `/docs` 문서에서 링크로 참조한다.
- 연구 단계에서 생성된 독립 실험 코드는 완전한 통합 시스템으로 가정하지 않고, 목적과 의존성을 명확히 적는다.
