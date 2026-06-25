## TCP
ros2 run tf2_ros static_transform_publisher 0.0 0.0 0.14 0.5 0.5 0.5 -0.5 tool0_controller tcp

## Opti-Track
### Default
ros2 run tf2_ros static_transform_publisher 0.225 0.325 0.0 0.5 -0.5 -0.5 0.5 base_link opti_world
### Custom
ros2 run tf2_ros static_transform_publisher 0.225 0.35 0.0 0.5 -0.5 -0.5 0.5 base_link opti_world

## Lucid Vision
<!-- ros2 run tf2_ros static_transform_publisher 0 0 0 0.5 -0.5 0.5 -0.5 triton_camera_link triton_camera -->
ros2 run tf2_ros static_transform_publisher 0.004061 0.05116 0.002374 0.02098779 0.00122301 -0.00032638 0.99977891 triton_camera helios_camera
ros2 run tf2_ros static_transform_publisher 0.3480980129049314 1.1366173565306177 0.3985054952068591 0.9315636603643267 9.153268858934325e-05 -0.0034850110724279895 0.36356153950632525 opti_world triton_camera


