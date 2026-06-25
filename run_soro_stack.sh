#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/min/project_SORO}"
BASHRC_SETTLE_SECONDS="${BASHRC_SETTLE_SECONDS:-2}"
START_GAP_SECONDS="${START_GAP_SECONDS:-2}"
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
CONDA_ROOT="${CONDA_ROOT:-/home/min/miniconda3}"
GENESIS_ENV_NAME="${GENESIS_ENV_NAME:-genesis}"
GENESIS_PYTHON="${GENESIS_PYTHON:-$CONDA_ROOT/envs/$GENESIS_ENV_NAME/bin/python}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

terminal_cmd_prefix() {
  cat <<EOF
cd "$PROJECT_DIR"
export PROJECT_DIR="$PROJECT_DIR"
export CONDA_ROOT="$CONDA_ROOT"
export GENESIS_ENV_NAME="$GENESIS_ENV_NAME"
export GENESIS_PYTHON="$GENESIS_PYTHON"
export ROS_DISTRO_NAME="$ROS_DISTRO_NAME"
source ~/.bashrc
sleep "$BASHRC_SETTLE_SECONDS"
EOF
}

ros_setup_suffix() {
  cat <<EOF
conda deactivate >/dev/null 2>&1 || true
if [ -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]; then
  source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
fi
if [ -f "$PROJECT_DIR/install/setup.bash" ]; then
  source "$PROJECT_DIR/install/setup.bash"
fi
EOF
}

hold_open_suffix() {
  cat <<'EOF'
status=$?
echo
echo "Process exited with status ${status}."
read -r -p "Press Enter to close this terminal..."
exit "$status"
EOF
}

launch_terminal() {
  local title="$1"
  local body="$2"
  local command

  command="$(terminal_cmd_prefix)
$body
$(hold_open_suffix)"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="$title" -- bash -lc "$command"
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -T "$title" -e bash -lc "$command"
  elif command -v konsole >/dev/null 2>&1; then
    konsole --new-tab --title "$title" -e bash -lc "$command"
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "$title" -e bash -lc "$command" &
  else
    echo "No supported terminal emulator found." >&2
    echo "Install gnome-terminal, x-terminal-emulator, konsole, or xterm." >&2
    exit 1
  fi
}

launch_ros_terminal() {
  local title="$1"
  local body="$2"
  launch_terminal "$title" "$(ros_setup_suffix)
$body"
}

echo "Launching SORO stack from $PROJECT_DIR"
echo "Bashrc settle interval: ${BASHRC_SETTLE_SECONDS}s"
echo

launch_ros_terminal \
  "SORO rosbridge" \
  'ros2 launch rosbridge_server rosbridge_websocket_launch.xml'
sleep "$START_GAP_SECONDS"

launch_terminal \
  "SORO Genesis scene" \
  'if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$GENESIS_ENV_NAME" >/dev/null 2>&1 || true
fi
if [ -x "$GENESIS_PYTHON" ]; then
  "$GENESIS_PYTHON" /home/min/project_SORO/src/genesis/hybrid_soft_robot_ros_scene.py
elif command -v python >/dev/null 2>&1; then
  python /home/min/project_SORO/src/genesis/hybrid_soft_robot_ros_scene.py
else
  echo "Genesis python not found. Set GENESIS_PYTHON=/path/to/python." >&2
  exit 127
fi'
sleep "$START_GAP_SECONDS"

launch_ros_terminal \
  "SORO camera pose GUI" \
  '/usr/bin/python3 /home/min/project_SORO/src/genesis/camera_pose_gui_publisher.py'
sleep 0.5

launch_ros_terminal \
  "SORO joint state GUI" \
  '/usr/bin/python3 /home/min/project_SORO/src/genesis/joint_state_gui_publisher.py'
sleep 0.5

launch_ros_terminal \
  "SORO RViz2" \
  'rviz2 -d default.rviz'
sleep 0.5

launch_ros_terminal \
  "SORO vision centerline" \
  '/usr/bin/python3 /home/min/project_SORO/src/vision_estimation/ros_centerline_client.py'

echo "Launch commands were sent to independent terminals."
