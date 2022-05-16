HOST_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ROS 2
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://${HOST_SETUP_DIR}/cyclonedds.xml
export ROS_DOMAIN_ID=69
export ROS_LOCALHOST_ONLY=1

# Ignition
export IGN_RELAY=127.0.0.1
export IGN_IP=127.0.0.1
