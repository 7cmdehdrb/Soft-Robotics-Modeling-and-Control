# ros_white_centerline_client.py 구조 설명

이 문서는 `ros_white_centerline_client.py`의 전체 구조와, 특히 `PointCloud2`를 이용해 2D 중심선을 3D 형상으로 복원하는 과정을 설명한다. 이 프로그램은 ROS2 환경에서 카메라 RGB 이미지와 깊이 정렬 포인트클라우드를 동시에 사용하여 흰색 소프트 로봇의 중심선을 실시간으로 추정하고, 그 결과를 3D marker와 pose array로 발행한다.

## 1. 프로그램의 목적

`ros_white_centerline_client.py`는 다음 순서로 동작한다.

1. ROS2 이미지 토픽에서 RGB/BGR 이미지를 수신한다.
2. 이미지에서 흰색 로봇 영역을 HSV 기준으로 분리한다.
3. `app.py`의 `estimate_centerline_from_image()`를 이용해 2D 중심선 픽셀 좌표를 추정한다.
4. 같은 장면에 대해 정렬된 `PointCloud2` 토픽을 수신한다.
5. 중심선 픽셀 좌표를 포인트클라우드의 `(u, v)` 좌표로 변환한다.
6. 각 `(u, v)` 위치에서 `(x, y, z)` 값을 읽어 3D 중심선으로 복원한다.
7. 복원된 3D 중심선을 `PoseArray`와 `MarkerArray`로 발행하여 RViz 등에서 확인할 수 있게 한다.

즉, 이 코드는 2D vision 기반 중심선 검출 결과를 depth camera의 organized point cloud와 결합해 소프트 로봇의 공간상 3D shape를 복원하는 ROS2 노드이다.

## 2. 주요 입출력 토픽

### 입력 토픽

- `--topic`
  - 기본값: `/camera/camera/color/image_raw`
  - RGB 또는 BGR 이미지 입력이다.
  - 흰색 로봇 segmentation과 2D centerline 추정에 사용된다.

- `--pointcloud-topic`
  - 기본값: `/camera/camera/depth/color/points`
  - depth가 color frame에 정렬된 `PointCloud2` 입력이다.
  - 2D 중심선 픽셀을 3D 좌표로 변환하는 데 사용된다.

### 출력 토픽

- `--marker-topic`
  - 기본값: `/white_centerline_3d/markers`
  - 3D 중심선 점과 선을 RViz marker로 발행한다.

- `--pose-array-topic`
  - 기본값: `/white_centerline_3d/poses`
  - 유효한 3D 중심선 점들을 `PoseArray`로 발행한다.

## 3. 전체 클래스 구조

핵심 클래스는 `WhiteImageCenterlineClient`이다. 이 클래스는 `rclpy.node.Node`를 상속하는 ROS2 노드이며, 이미지와 포인트클라우드를 구독하고 결과를 발행한다.

### `__init__()`

초기화 단계에서는 다음 작업을 수행한다.

- 이미지 토픽 구독자를 생성한다.
- 포인트클라우드 토픽 구독자를 생성한다.
- 3D marker publisher를 생성한다.
- 3D pose array publisher를 생성한다.
- `process_hz` 주기로 `_process_latest()`가 실행되도록 timer를 만든다.
- `--show` 옵션이 켜져 있으면 matplotlib debug window를 준비한다.

이미지와 포인트클라우드는 서로 다른 callback에서 들어오기 때문에, 최신 데이터를 안전하게 공유하기 위해 `threading.Lock`을 사용한다.

### `_image_callback()`

이미지 메시지를 받을 때마다 실행된다.

- `ros_image_to_bgr()`로 ROS `Image` 메시지를 OpenCV BGR 이미지로 변환한다.
- 최신 이미지와 이미지 timestamp를 저장한다.
- 새 이미지가 들어온 시각을 `_latest_stamp`에 기록한다.

지원하는 이미지 encoding은 다음과 같다.

- `rgb8`
- `bgr8`
- `rgba8`
- `bgra8`
- `mono8`
- `8uc1`

### `_pointcloud_callback()`

`PointCloud2` 메시지를 받을 때마다 실행된다.

- 최신 포인트클라우드를 `_latest_cloud`에 저장한다.
- 수신 시각을 `_latest_cloud_stamp`에 저장한다.

이 timestamp는 이후 이미지 처리 시점에서 포인트클라우드가 너무 오래된 데이터인지 확인하는 데 사용된다.

### `_process_latest()`

주기적으로 최신 이미지를 처리하는 핵심 루프이다.

처리 순서는 다음과 같다.

1. 아직 이미지가 없으면 반환한다.
2. 이미 처리한 이미지이면 다시 처리하지 않는다.
3. 최신 이미지와 최신 포인트클라우드를 복사한다.
4. `estimate_centerline_from_image()`로 2D 중심선을 계산한다.
5. debug grid를 만든다.
6. 포인트클라우드가 존재하고 `max_cloud_age`보다 오래되지 않았으면 3D 복원을 시도한다.
7. 복원에 성공하면 `_publish_centerline_3d()`로 결과를 발행한다.
8. 상태 로그를 주기적으로 출력한다.
9. `--show`가 활성화되어 있으면 debug grid를 화면에 표시한다.

## 4. 2D 중심선 추정 흐름

2D 중심선 계산은 이 파일 내부에서 직접 구현하지 않고 `app.py`의 함수를 사용한다.

```python
result = estimate_centerline_from_image(
    image_bgr=image_bgr,
    hsv_range=self.hsv_range,
    backbone_points=self.args.points,
    epochs=self.args.epochs,
    alpha0=self.args.alpha,
    radius0=self.args.radius,
    canny_low=self.args.canny_low,
    canny_high=self.args.canny_high,
)
```

이 함수의 결과인 `result.som_points`가 2D 중심선 좌표이다. 각 점은 이미지 좌표계의 `(x, y)` 또는 `(u, v)` 픽셀 위치를 의미한다.

기본 HSV 범위는 흰색 로봇을 검출하도록 설정되어 있다.

```python
WHITE_ROBOT_HSV = HSVRange((0, 0, 150), (179, 80, 255))
```

필요하면 실행 인자로 `--hsv-lower`, `--hsv-upper`를 제공해 조명 환경에 맞게 조정할 수 있다.

## 5. PointCloud를 통한 3D Shape 복원 핵심 아이디어

이 프로그램에서 3D shape 복원은 전체 물체 표면을 mesh로 복원하는 방식이 아니라, 2D 이미지에서 찾은 중심선 backbone을 3D 공간의 점열로 lifting하는 방식이다. 결과적으로 소프트 로봇의 3D centerline shape가 복원된다.

핵심 함수는 다음이다.

```python
points_3d = centerline_pixels_to_3d(
    cloud=cloud,
    pixels_xy=result.som_points,
    image_shape=image_bgr.shape[:2],
    search_radius=self.args.point_search_radius,
    allow_zero_points=self.args.allow_zero_points,
)
```

여기서 입력과 출력은 다음과 같다.

- 입력 `cloud`: depth camera가 제공하는 `PointCloud2`
- 입력 `pixels_xy`: 2D 중심선 픽셀 좌표 배열
- 입력 `image_shape`: RGB 이미지의 높이와 너비
- 출력 `points_3d`: 각 중심선 픽셀에 대응하는 `(x, y, z)` 좌표 배열

## 6. Organized PointCloud2가 필요한 이유

`centerline_pixels_to_3d()`는 포인트클라우드를 이미지처럼 `(u, v)`로 접근한다. 따라서 입력 `PointCloud2`는 organized point cloud여야 한다.

코드에서는 다음 조건을 검사한다.

```python
if cloud.height == 1 and image_height > 1:
    raise RuntimeError(
        "PointCloud2 is not organized; enable an ordered/aligned point cloud."
    )
```

`cloud.height == 1`인 포인트클라우드는 일반적으로 unorganized point cloud이다. 이런 데이터는 점들의 1차원 목록일 뿐이므로 특정 이미지 픽셀 `(u, v)`에 해당하는 점을 바로 찾을 수 없다.

이 프로그램은 `/camera/camera/depth/color/points`처럼 depth가 color image에 정렬된 ordered/aligned point cloud를 전제로 한다. 이 조건이 만족되면 RGB 이미지의 픽셀 좌표와 포인트클라우드의 픽셀 좌표가 같은 장면 위치를 가리키게 된다.

## 7. 2D 픽셀에서 3D 좌표로 변환하는 과정

### 7.1 이미지 좌표를 포인트클라우드 좌표로 스케일링

RGB 이미지 크기와 포인트클라우드 크기가 항상 같다고 가정할 수 없기 때문에 좌표 스케일링을 수행한다.

```python
u = scale_pixel_coordinate(pixel[0], image_width, cloud.width)
v = scale_pixel_coordinate(pixel[1], image_height, cloud.height)
```

`scale_pixel_coordinate()`는 다음 식과 같은 의미를 갖는다.

```text
cloud_coordinate = image_coordinate * (cloud_size - 1) / (image_size - 1)
```

그리고 결과를 반올림한 뒤 유효 범위 `[0, cloud_size - 1]`로 clamp한다. 이를 통해 이미지 중심선 점이 포인트클라우드의 가장 가까운 픽셀 위치로 대응된다.

### 7.2 PointCloud2 binary buffer에서 XYZ 추출

`PointCloud2` 메시지는 각 점의 데이터를 binary buffer인 `cloud.data`에 저장한다. 따라서 `(u, v)` 위치의 XYZ를 읽으려면 다음 정보가 필요하다.

- `row_step`: 한 행이 차지하는 byte 수
- `point_step`: 한 점이 차지하는 byte 수
- field offset: `x`, `y`, `z`가 각 점 내부에서 시작되는 byte 위치
- endian 정보: big endian인지 little endian인지

이 역할을 `PointCloudXYZUnpacker`가 담당한다.

```python
base = int(v * cloud.row_step + u * cloud.point_step)
```

위 식은 `(u, v)` 점이 `cloud.data` 안에서 시작되는 byte 위치를 계산한다. 이후 `x`, `y`, `z` 각각의 offset을 더해 float32 값을 읽는다.

```python
return np.array(
    [
        self.structs[i].unpack_from(data, base + self.offsets[i])[0]
        for i in range(3)
    ],
    dtype=np.float32,
)
```

즉, `PointCloud2`의 organized 2D 배열 인덱스 `(u, v)`를 binary memory offset으로 바꾼 뒤, 그 위치에서 실제 3D 좌표를 복원한다.

### 7.3 유효하지 않은 depth에 대한 근방 탐색

Depth camera에서는 반사, occlusion, segmentation 경계, 측정 실패 때문에 특정 픽셀의 3D 값이 `NaN`, `Inf`, 또는 `(0, 0, 0)`일 수 있다. 코드에서는 먼저 중심선 픽셀의 정확한 위치에서 XYZ를 읽고, 유효하지 않으면 주변 픽셀을 탐색한다.

```python
point = unpacker.xyz_at(cloud, u, v)
if is_valid_xyz(point, allow_zero_points):
    return point
```

정확한 위치가 유효하지 않으면 `search_radius` 범위 안에서 가장 가까운 유효 포인트를 찾는다.

```python
for dv in range(-radius, radius + 1):
    for du in range(-radius, radius + 1):
        distance = du * du + dv * dv
```

가장 작은 `du^2 + dv^2`를 갖는 유효한 point를 선택하므로, 중심선 픽셀에 가장 가까운 depth 값을 사용하게 된다. 이 방식은 얇은 소프트 로봇의 중심선이 depth hole 근처에 걸리는 경우에도 3D 중심선이 끊기지 않도록 도와준다.

### 7.4 유효한 3D 점 판정

`is_valid_xyz()`는 다음 조건을 확인한다.

```python
if not np.isfinite(point).all():
    return False
if not allow_zero_points and np.linalg.norm(point) < 1e-9:
    return False
return True
```

기본적으로 `NaN`, `Inf`는 무효이다. 또한 `--allow-zero-points`를 사용하지 않으면 원점에 가까운 `(0, 0, 0)` 값도 무효로 처리한다. 많은 depth pipeline에서 측정 실패가 0으로 표현될 수 있기 때문이다.

## 8. 3D 중심선 발행 방식

3D 복원이 끝나면 `_publish_centerline_3d()`가 호출된다.

### 8.1 PoseArray 발행

각 유효한 3D point는 `geometry_msgs/Pose`로 변환된다.

- position: 복원된 `(x, y, z)`
- orientation: `w = 1.0`인 identity quaternion

이 pose들이 `PoseArray`에 담겨 `/white_centerline_3d/poses`로 발행된다.

### 8.2 MarkerArray 발행

RViz 시각화를 위해 `MarkerArray`도 발행한다.

- `SPHERE` marker: 각 3D 중심선 점을 구 형태로 표시
- `LINE_STRIP` marker: 유효한 중심선 점들을 선으로 연결
- `DELETEALL` marker: 이전 marker를 지우고 최신 결과만 남기기 위함

이 구조 덕분에 RViz에서는 소프트 로봇의 3D centerline이 점과 선으로 동시에 표시된다.

## 9. 시간 동기화와 stale cloud 처리

이미지와 포인트클라우드는 별도의 subscriber에서 수신된다. 이 코드는 엄밀한 message filter synchronization을 사용하지 않고, 최신 이미지와 최신 포인트클라우드를 저장해 두었다가 처리 시점에 결합한다.

이를 보완하기 위해 `cloud_age`를 계산한다.

```python
cloud_age = time.time() - self._latest_cloud_stamp
```

그리고 다음 조건을 만족할 때만 3D 복원을 수행한다.

```python
cloud is not None and cloud_age <= self.args.max_cloud_age
```

기본 `--max-cloud-age`는 `0.5`초이다. 포인트클라우드가 너무 오래되면 `stale_cloud` 상태를 로그에 남기고 3D 복원은 건너뛴다.

## 10. 실행 인자 요약

3D 복원과 직접 관련된 주요 인자는 다음과 같다.

- `--pointcloud-topic`
  - organized/aligned `PointCloud2` 입력 토픽

- `--max-cloud-age`
  - 이미지 처리 시 사용할 수 있는 포인트클라우드의 최대 허용 age

- `--point-search-radius`
  - 중심선 픽셀 위치의 depth가 무효일 때 주변에서 유효 point를 찾는 반경

- `--allow-zero-points`
  - `(0, 0, 0)`에 가까운 point를 유효한 point로 허용할지 여부

- `--marker-diameter`
  - RViz sphere marker 크기

- `--line-width`
  - RViz line strip marker 두께

- `--marker-lifetime`
  - marker 유지 시간

2D 중심선 검출과 관련된 주요 인자는 다음과 같다.

- `--hsv-lower`, `--hsv-upper`
- `--points`
- `--epochs`
- `--alpha`
- `--radius`
- `--canny-low`, `--canny-high`

## 11. 데이터 흐름 요약

```text
ROS Image
  -> ros_image_to_bgr()
  -> HSV white segmentation + contour/centerline estimation
  -> 2D centerline pixels: result.som_points

ROS PointCloud2
  -> PointCloudXYZUnpacker
  -> organized cloud pixel access: (u, v)
  -> XYZ sampling with local fallback search

2D centerline pixels + PointCloud2
  -> centerline_pixels_to_3d()
  -> 3D centerline points
  -> PoseArray + MarkerArray
  -> RViz / downstream controller
```

## 12. 3D Shape 복원 관점에서의 의미

이 프로그램의 3D shape 복원은 다음 가정을 기반으로 한다.

1. RGB image와 depth point cloud가 같은 카메라 기준으로 정렬되어 있다.
2. 흰색 소프트 로봇의 중심선이 2D 이미지에서 안정적으로 추정된다.
3. 각 중심선 픽셀 주변에 유효한 depth point가 존재한다.
4. 중심선 점들의 순서가 로봇 backbone의 순서를 나타낸다.

이 조건이 만족되면 2D에서 얻은 centerline backbone은 포인트클라우드의 XYZ 값을 통해 3D backbone으로 변환된다. 따라서 출력된 3D 점열은 소프트 로봇의 굽힘 상태, 위치, 자세를 나타내는 간결한 shape representation으로 사용할 수 있다.

제어 관점에서는 전체 표면 mesh보다 중심선 backbone이 더 유용할 수 있다. 소프트 로봇의 bending, tip position, curvature, segment direction 등을 계산하기 쉽고, downstream controller나 estimator가 처리해야 할 데이터 크기도 작기 때문이다.

## 13. 주의할 점과 개선 가능성

- 현재 코드는 엄밀한 timestamp synchronization을 사용하지 않는다. 빠른 움직임에서는 `message_filters`를 이용한 image-pointcloud 동기화가 더 안정적일 수 있다.
- point cloud가 반드시 organized/aligned 형태여야 한다. unorganized point cloud에서는 현재 방식으로 픽셀 기반 lookup을 수행할 수 없다.
- `search_radius`가 너무 작으면 depth hole에서 중심선이 끊길 수 있고, 너무 크면 주변 배경 depth를 잘못 선택할 수 있다.
- 흰색 segmentation은 조명에 민감하므로 환경에 따라 HSV 범위 조정이 필요하다.
- 복원된 3D 중심선에 smoothing이나 temporal filtering을 추가하면 marker jitter를 줄일 수 있다.
