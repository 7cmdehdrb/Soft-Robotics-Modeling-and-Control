```bash
ros2 launch realsense2_camera rs_launch.py \
  pointcloud.enable:=true \
  pointcloud.ordered_pc:=true \
  align_depth.enable:=true \
  enable_sync:=true \
  enable_gyro:=true \
  enable_accel:=true
```


```bash
ros2 param set /camera/camera pointcloud.enable true
ros2 param set /camera/camera pointcloud.ordered_pc true
ros2 param set /camera/camera align_depth.enable true
ros2 param set /camera/camera enable_sync true
```

```bash
ros2 param get /camera/camera pointcloud.enable
ros2 param get /camera/camera pointcloud.ordered_pc
ros2 param get /camera/camera align_depth.enable
ros2 param get /camera/camera enable_sync
```