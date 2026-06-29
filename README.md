# Soft Robotics Modeling and Control

이 레포지터리는 소프트 로봇의 모델링, 센싱, 상태 추정, 제어, 시각화를 실험하기 위한 연구용 작업 공간이다. 아직 리서치 단계이므로 패키지들이 하나의 완성된 시스템으로 완전히 통합되어 있지는 않으며, 각 폴더는 특정 장비·모델·실험 목적을 중심으로 독립적으로 발전하고 있다.

## 문서 시작점

전체 문서는 [`docs/`](docs/) 아래에 정리되어 있다.

| 문서 | 설명 |
| --- | --- |
| [`docs/README.md`](docs/README.md) | 문서 허브와 패키지별 문서 링크 |
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | 프로젝트 전체 목표와 연구 단계 구성 설명 |
| [`docs/DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md) | 레포지터리 디렉토리 구조 설명 |
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | RealSense/ROS2 실행 명령 모음 |
| [`docs/tf.md`](docs/tf.md) | TF static transform publisher 명령 메모 |

## 핵심 방향

현재 프로젝트의 큰 흐름은 다음과 같다.

1. 소프트 로봇에 안전한 범위의 actuation input을 인가한다.
2. 센서, OptiTrack, camera/depth, 기타 실험 장비에서 데이터를 수집한다.
3. 입력/출력 데이터를 시간 기준으로 정렬하고 로깅한다.
4. LSTM 등 학습 기반 모델 또는 Cosserat rod와 같은 해석 모델로 상태를 추정한다.
5. ROS2 토픽과 RViz를 통해 결과를 시각화하고 제어 실험으로 확장한다.

패키지별 상세 설명은 각 패키지 내부 Markdown을 유지하고, [`docs/README.md`](docs/README.md)에서 링크로 참조한다.
