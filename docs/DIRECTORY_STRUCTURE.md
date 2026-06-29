# 디렉토리 구조 설명

이 문서는 레포지터리의 주요 디렉토리와 파일 역할을 설명한다. 현재 레포지터리는 연구용 작업 공간이므로, 모든 폴더가 하나의 완성된 애플리케이션으로 통합되어 있다고 가정하지 않는다.

## 1. 최상위 구조

```text
.
├── README.md
├── COMMAND.md
├── docs/
├── src/
└── test/
```

| 경로 | 설명 |
| --- | --- |
| `README.md` | 레포지터리 진입점. 상세 문서는 `/docs`로 연결한다. |
| `COMMAND.md` | 기존 ROS2/RealSense 명령 메모. 통합 문서는 `docs/COMMANDS.md`에 정리했다. |
| `docs/` | 프로젝트 전체 설명, 디렉토리 구조, 공통 명령, TF 메모를 담는 상위 문서 폴더 |
| `src/` | ROS2 패키지, 실험 스크립트, 모델링 코드, 하드웨어 연동 코드가 모여 있는 작업 폴더 |
| `test/` | 최상위 테스트용 폴더. 현재 주요 패키지 테스트는 각 ROS2 패키지의 `test/`에 더 가깝다. |

## 2. `/docs`

```text
docs/
├── README.md
├── PROJECT_OVERVIEW.md
├── DIRECTORY_STRUCTURE.md
├── COMMANDS.md
└── tf.md
```

| 파일 | 설명 |
| --- | --- |
| `docs/README.md` | 문서 허브와 패키지별 Markdown 링크 |
| `docs/PROJECT_OVERVIEW.md` | 프로젝트 전체 목적과 주요 서브시스템 설명 |
| `docs/DIRECTORY_STRUCTURE.md` | 레포지터리 디렉토리 구조 설명 |
| `docs/COMMANDS.md` | RealSense, point cloud, ROS2 parameter 관련 실행 명령 정리 |
| `docs/tf.md` | static transform publisher 명령 메모 |

## 3. `/src` 주요 폴더

```text
src/
├── CR/
├── ROS-TCP-Endpoint-ROS2v0.7.0/
├── arduino/
├── genesis/
├── lstm/
├── natnet/
├── psf_control/
└── vision_estimation/
```

| 경로 | 성격 | 관련 문서 |
| --- | --- | --- |
| `src/CR/` | Cosserat rod 기반 소프트 로봇 모델링 및 GUI 실험 | [`../src/CR/GEOMETRIC_PARAM.md`](../src/CR/GEOMETRIC_PARAM.md), [`../src/CR/PHYSICS_PARAM.md`](../src/CR/PHYSICS_PARAM.md) |
| `src/ROS-TCP-Endpoint-ROS2v0.7.0/` | Unity ROS-TCP Endpoint 외부 패키지 | [`../src/ROS-TCP-Endpoint-ROS2v0.7.0/README.md`](../src/ROS-TCP-Endpoint-ROS2v0.7.0/README.md) |
| `src/arduino/` | Arduino 공압 제어 firmware와 serial sender | [`../src/arduino/README.md`](../src/arduino/README.md) |
| `src/genesis/` | Genesis simulation 및 ROS 연동 실험 스크립트 | 아직 별도 문서 없음 |
| `src/lstm/` | LSTM 기반 마커 위치 예측 모델 학습 ROS2 패키지/코드 | [`../src/lstm/README.md`](../src/lstm/README.md) |
| `src/natnet/` | OptiTrack Motive NatNet 데이터를 ROS2로 발행하는 패키지 | [`../src/natnet/README.md`](../src/natnet/README.md) |
| `src/psf_control/` | 공압 soft finger 제어와 LSTM predictor ROS2 패키지 | [`../src/psf_control/README.md`](../src/psf_control/README.md) |
| `src/vision_estimation/` | 이미지/depth 기반 centerline 및 shape estimation 실험 | [`../src/vision_estimation/VISION.md`](../src/vision_estimation/VISION.md), [`../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md`](../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md) |

## 4. ROS2 패키지로 보이는 폴더

다음 폴더는 `package.xml`, `setup.py`, `setup.cfg` 등을 포함하므로 ROS2 Python 패키지로 빌드/실행하는 구조에 가깝다.

| 패키지 | 주요 역할 |
| --- | --- |
| `src/lstm` | LSTM 학습/추론 관련 코드 |
| `src/natnet` | NatNet client와 ROS2 publisher |
| `src/psf_control` | LSTM predictor와 soft finger 제어 관련 node |
| `src/ROS-TCP-Endpoint-ROS2v0.7.0` | Unity 연동 endpoint |

단, 연구 단계 특성상 `package.xml`이 있다고 해서 모든 코드가 같은 launch graph로 통합되어 있다는 의미는 아니다.

## 5. 실험 스크립트 중심 폴더

다음 폴더는 ROS2 패키지라기보다 실험 파일과 단일 실행 스크립트가 중심이다.

| 폴더 | 주요 파일/역할 |
| --- | --- |
| `src/arduino` | `pneumatic_pid_control.ino`, `send_pneumatic_control.py` |
| `src/CR` | Cosserat rod GUI Python scripts, geometry/physics parameter notes |
| `src/genesis` | simulation examples, GUI publisher scripts |
| `src/vision_estimation` | `app.py`, `ros_centerline_client.py`, `ros_white_centerline_client.py`, output folders |

## 6. 문서 배치 원칙

- `/docs`는 레포지터리 전체 관점의 문서를 담는다.
- `src/<package>/README.md`와 같은 패키지 내부 문서는 해당 패키지를 이해하기 위한 문서이므로 이동하지 않는다.
- `/docs` 문서는 패키지 내부 문서를 링크로 참조한다.
- 실험 메모 성격의 문서도 가능한 한 `/docs`에서 인덱싱하여 찾기 쉽게 한다.

## 7. 권장 탐색 순서

처음 레포지터리를 보는 경우 다음 순서로 읽는 것을 권장한다.

1. [`docs/PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)
2. [`docs/DIRECTORY_STRUCTURE.md`](DIRECTORY_STRUCTURE.md)
3. 관심 있는 패키지의 README 또는 세부 문서
4. [`docs/COMMANDS.md`](COMMANDS.md)와 [`docs/tf.md`](tf.md)에서 실험 명령 확인
