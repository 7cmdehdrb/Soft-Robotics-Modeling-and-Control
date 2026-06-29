# 실행 명령 모음

이 문서는 기존 `COMMAND.md`에 있던 RealSense/ROS2 명령과 `docs/tf.md`에 있는 TF 메모를 찾기 쉽게 연결한 공통 명령 문서이다. 명령은 실험 환경에 따라 달라질 수 있으므로, 실행 전 토픽 이름과 노드 이름을 확인해야 한다.

## 1. RealSense camera launch

ordered point cloud, depth-color alignment, gyro/accel, sync를 켠 상태로 RealSense camera를 실행하는 예시이다.

```bash
ros2 launch realsense2_camera rs_launch.py \
  pointcloud.enable:=true \
  pointcloud.ordered_pc:=true \
  align_depth.enable:=true \
  enable_sync:=true \
  enable_gyro:=true \
  enable_accel:=true
```

`vision_estimation`의 2D centerline → 3D 변환은 organized/aligned point cloud를 전제로 하므로, `pointcloud.ordered_pc`와 `align_depth.enable` 설정이 중요하다.

## 2. RealSense parameter set

이미 실행 중인 camera node에 대해 관련 parameter를 설정하는 예시이다.

```bash
ros2 param set /camera/camera pointcloud.enable true
ros2 param set /camera/camera pointcloud.ordered_pc true
ros2 param set /camera/camera align_depth.enable true
ros2 param set /camera/camera enable_sync true
```

## 3. RealSense parameter 확인

설정값을 확인하는 예시이다.

```bash
ros2 param get /camera/camera pointcloud.enable
ros2 param get /camera/camera pointcloud.ordered_pc
ros2 param get /camera/camera align_depth.enable
ros2 param get /camera/camera enable_sync
```

## 4. TF 명령

장비 간 static transform publisher 명령은 [`tf.md`](tf.md)에 정리되어 있다.

## 5. 관련 패키지 문서

- Vision 기반 point cloud 사용: [`../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md`](../src/vision_estimation/ROS_WHITE_CENTERLINE_CLIENT.md)
- NatNet/OptiTrack 토픽: [`../src/natnet/README.md`](../src/natnet/README.md)
- LSTM predictor 실행: [`../src/psf_control/README.md`](../src/psf_control/README.md)
