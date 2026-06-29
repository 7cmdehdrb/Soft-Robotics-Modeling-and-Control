# `ros_white_centerline_client.py` 설명

이 문서는 `src/vision_estimation/ros_white_centerline_client.py` 파일의 목적, ROS2 입출력, 처리 흐름, 주요 함수와 실행 옵션을 정리한다.

## 1. 파일의 목적

`ros_white_centerline_client.py`는 ROS2 노드로 실행되는 비전 기반 중심선 추정 클라이언트이다. 카메라 RGB 이미지에서 어두운 배경 위의 흰색 소프트 로봇을 분리하고, `app.py`의 중심선 추정 로직을 이용해 2D 중심선 점들을 계산한다. 동시에 정렬된 `PointCloud2` 깊이 정보를 사용해 각 2D 중심선 픽셀을 3D 좌표로 변환한 뒤, RViz에서 볼 수 있는 마커와 `PoseArray`로 발행한다.

전체 흐름은 다음과 같다.

```text
ROS2 Image 수신
  → OpenCV BGR 이미지로 변환
  → HSV threshold + contour + SOM 기반 2D 중심선 추정
  → 정렬된 PointCloud2에서 대응 3D 점 샘플링
  → MarkerArray와 PoseArray 발행
  → 선택적으로 matplotlib 디버그 화면 표시
```

## 2. 주요 의존 모듈

이 파일은 크게 네 종류의 라이브러리를 사용한다.

| 구분 | 사용 모듈 | 역할 |
| --- | --- | --- |
| 영상 처리 | `cv2`, `numpy`, `matplotlib` | 이미지 변환, 디버그 화면 표시, 배열 연산 |
| ROS2 | `rclpy`, `sensor_msgs`, `geometry_msgs`, `visualization_msgs` | 이미지/포인트클라우드 구독, 마커/포즈 발행 |
| 바이너리 파싱 | `struct` | `PointCloud2`의 `x`, `y`, `z` float32 필드 해석 |
| 프로젝트 내부 | `HSVRange`, `estimate_centerline_from_image`, `make_debug_grid`, `parse_hsv_triplet` | HSV 범위 정의, 중심선 추정, 디버그 이미지 생성, CLI HSV 파싱 |

기본 흰색 로봇 검출 범위는 다음과 같다.

```python
WHITE_ROBOT_HSV = HSVRange((0, 0, 150), (179, 80, 255))
```

즉, Hue는 전체 범위를 허용하고, Saturation은 낮으며, Value는 높은 픽셀을 흰색 로봇 후보로 본다.

## 3. ROS2 노드 구조

핵심 클래스는 `WhiteImageCenterlineClient`이다. 이 클래스는 `rclpy.node.Node`를 상속하며, 노드 이름은 `white_image_centerline_client`이다.

### 3.1 구독 토픽

| 토픽 옵션 | 기본값 | 메시지 타입 | 설명 |
| --- | --- | --- | --- |
| `--topic` | `/camera/camera/color/image_raw` | `sensor_msgs/Image` | 중심선 추정에 사용할 컬러 이미지 |
| `--pointcloud-topic` | `/camera/camera/depth/color/points` | `sensor_msgs/PointCloud2` | 2D 픽셀을 3D 좌표로 변환할 정렬된 포인트클라우드 |

포인트클라우드는 이미지와 픽셀 대응이 가능한 **organized/aligned point cloud**여야 한다. `PointCloud2.height == 1`인 비정렬 포인트클라우드는 2D 픽셀 위치와 직접 대응할 수 없으므로 오류로 처리된다.

### 3.2 발행 토픽

| 토픽 옵션 | 기본값 | 메시지 타입 | 설명 |
| --- | --- | --- | --- |
| `--marker-topic` | `/white_centerline_3d/markers` | `visualization_msgs/MarkerArray` | RViz 시각화용 구/선 마커 |
| `--pose-array-topic` | `/white_centerline_3d/poses` | `geometry_msgs/PoseArray` | 유효한 3D 중심선 점들의 pose 배열 |

## 4. 실행 중 데이터 흐름

### 4.1 이미지 콜백

`_image_callback()`은 ROS2 `Image` 메시지를 받으면 `ros_image_to_bgr()`로 OpenCV BGR 배열로 변환한다. 변환에 실패하면 경고 로그를 남기고 해당 프레임을 버린다.

변환에 성공하면 다음 데이터를 lock으로 보호하면서 저장한다.

- 최신 BGR 이미지
- 이미지 메시지의 timestamp
- 로컬 수신 시각

### 4.2 포인트클라우드 콜백

`_pointcloud_callback()`은 최신 `PointCloud2` 메시지와 로컬 수신 시각을 저장한다. 이후 이미지 처리 타이머에서 이 포인트클라우드가 너무 오래되지 않았는지 검사한다.

### 4.3 주기 처리 타이머

`_process_latest()`는 `--process-hz` 주기로 실행된다. 기본값은 5 Hz이다.

처리 순서는 다음과 같다.

1. 아직 이미지가 없으면 return
2. 이미 처리한 이미지 timestamp이면 return
3. 최신 이미지를 복사하고 최신 포인트클라우드를 가져옴
4. `estimate_centerline_from_image()`로 2D 중심선 추정
5. `make_debug_grid()`로 디버그 이미지 생성
6. 최신 포인트클라우드가 있고 `--max-cloud-age`보다 오래되지 않았으면 3D 변환 수행
7. 유효한 3D 점이 있으면 `MarkerArray`와 `PoseArray` 발행
8. `--log-period` 주기로 상태 로그 출력
9. `--show`가 켜져 있으면 matplotlib 창에 디버그 그리드 표시

## 5. 2D 중심선 추정

2D 중심선 추정 자체는 이 파일에서 직접 구현하지 않고 `app.py`의 `estimate_centerline_from_image()`에 위임한다. 이 함수에 전달되는 주요 파라미터는 다음과 같다.

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--points` | `7` | 추정할 중심선 대표점 개수 |
| `--epochs` | `4` | SOM 학습 반복 수 |
| `--alpha` | `0.01` | SOM 초기 학습률 |
| `--radius` | `3.0` | SOM neighborhood radius |
| `--canny-low` | `50` | Canny edge lower threshold |
| `--canny-high` | `150` | Canny edge upper threshold |
| `--hsv-lower` | 없음 | 수동 HSV 하한값, 예: `0,0,150` |
| `--hsv-upper` | 없음 | 수동 HSV 상한값, 예: `179,80,255` |

`--hsv-lower`와 `--hsv-upper`는 반드시 함께 지정해야 한다. 둘 중 하나만 지정하면 `main()`에서 `ValueError`가 발생한다.

## 6. ROS Image → BGR 변환

`ros_image_to_bgr()`는 ROS2 `Image` 메시지를 OpenCV가 사용하는 BGR 배열로 변환한다.

지원하는 encoding은 다음과 같다.

- `rgb8`
- `bgr8`
- `rgba8`
- `bgra8`
- `mono8`
- `8uc1`

각 encoding에서 `height`, `width`, `step`, `data` 크기를 검사해 메시지 데이터가 부족하면 `ValueError`를 발생시킨다. `rgb8`, `rgba8`, `bgra8`, grayscale 계열은 OpenCV 색상 변환을 거쳐 BGR 3채널 이미지로 통일된다.

## 7. 2D 픽셀 → 3D 좌표 변환

`centerline_pixels_to_3d()`는 중심선 픽셀 좌표 배열과 정렬된 `PointCloud2`를 받아 3D 점 배열을 만든다.

### 7.1 입력 검증

다음 조건을 만족하지 않으면 `RuntimeError`가 발생한다.

- 포인트클라우드의 `height`, `width`가 양수여야 함
- 이미지가 2D인데 포인트클라우드가 `height == 1`인 비정렬 구조이면 안 됨
- `row_step`, `point_step`이 양수여야 함
- `PointCloud2`에 `x`, `y`, `z` 필드가 있어야 함
- `x`, `y`, `z` 필드는 모두 `FLOAT32`여야 함

### 7.2 좌표 스케일링

이미지 해상도와 포인트클라우드 해상도가 다를 수 있으므로, `scale_pixel_coordinate()`가 이미지 픽셀 좌표를 포인트클라우드의 `u`, `v` 좌표로 선형 스케일링한다.

### 7.3 주변 유효점 검색

`sample_cloud_xyz()`는 먼저 해당 `u`, `v` 위치의 `x`, `y`, `z`를 읽는다. 점이 유효하지 않으면 `--point-search-radius` 범위 안에서 가장 가까운 유효점을 찾는다.

유효점 판단 기준은 `is_valid_xyz()`에 있다.

- `NaN` 또는 `inf`가 없어야 함
- `--allow-zero-points`가 꺼져 있으면 원점에 가까운 `(0, 0, 0)` 점은 무효로 처리

## 8. 3D 결과 발행

`_publish_centerline_3d()`는 유효한 3D 점을 두 가지 형태로 발행한다.

### 8.1 `PoseArray`

각 3D 점은 position으로 들어가며, orientation은 단위 quaternion `w=1.0`으로 설정된다. `NaN`이 포함된 점은 제외된다.

### 8.2 `MarkerArray`

RViz 시각화를 위해 다음 마커를 생성한다.

1. `DELETEALL` 마커로 이전 중심선 마커 제거
2. 각 중심선 점을 주황색 `SPHERE` 마커로 표시
3. 유효점이 2개 이상이면 청록색 `LINE_STRIP` 마커로 중심선 연결

마커 크기와 lifetime은 CLI 옵션으로 조절한다.

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--marker-diameter` | `0.012` | 중심선 점 구 마커 지름 |
| `--line-width` | `0.006` | 중심선 line strip 두께 |
| `--marker-lifetime` | `0.5` | 마커 유지 시간, 초 단위 |

## 9. 디버그 표시와 실패 처리

`--show`는 기본적으로 켜져 있으며, matplotlib 창에 디버그 그리드를 표시한다. GUI가 없는 환경이나 원격 실행 환경에서는 다음처럼 끌 수 있다.

```bash
python3 src/vision_estimation/ros_white_centerline_client.py --no-show
```

중심선 추정 중 `RuntimeError`가 발생하면 `make_failure_grid()`가 원본 이미지 위쪽에 실패 메시지를 빨간색으로 표시한 디버그 이미지를 만든다.

## 10. 주요 실행 옵션 예시

기본 토픽으로 실행하되 GUI를 끄는 예시는 다음과 같다.

```bash
python3 src/vision_estimation/ros_white_centerline_client.py --no-show
```

HSV 범위를 수동으로 지정하는 예시는 다음과 같다.

```bash
python3 src/vision_estimation/ros_white_centerline_client.py \
  --hsv-lower 0,0,150 \
  --hsv-upper 179,80,255 \
  --points 7 \
  --process-hz 5 \
  --no-show
```

다른 카메라 토픽과 포인트클라우드 토픽을 사용하는 예시는 다음과 같다.

```bash
python3 src/vision_estimation/ros_white_centerline_client.py \
  --topic /camera/color/image_raw \
  --pointcloud-topic /camera/depth/color/points \
  --marker-topic /white_centerline_3d/markers \
  --pose-array-topic /white_centerline_3d/poses \
  --no-show
```

## 11. 주의할 점

- 포인트클라우드는 컬러 이미지와 정렬되어 있어야 2D 픽셀에서 3D 좌표를 안정적으로 읽을 수 있다.
- `--max-cloud-age` 기본값은 `0.5`초이므로, 포인트클라우드 수신이 느리거나 끊기면 3D 발행이 중단되고 stale 상태가 로그에 표시된다.
- 흰색 물체 검출은 조명과 배경에 민감하다. 환경이 바뀌면 `--hsv-lower`, `--hsv-upper`, Canny threshold를 조정해야 한다.
- `--points`는 최소 2 이상이어야 한다.
- GUI가 없는 ROS 실행 환경에서는 `--no-show`를 사용하는 것이 안전하다.

## 12. 요약

이 파일은 RGB 이미지 기반 2D 중심선 추정과 정렬된 depth point cloud 기반 3D 복원을 연결하는 ROS2 실시간 노드이다. 핵심 역할은 다음 세 가지로 요약된다.

1. ROS2 이미지에서 흰색 소프트 로봇의 2D 중심선을 추정한다.
2. 정렬된 `PointCloud2`를 이용해 중심선 픽셀을 3D 좌표로 변환한다.
3. RViz와 다른 ROS2 노드에서 사용할 수 있도록 `MarkerArray`와 `PoseArray`를 발행한다.
