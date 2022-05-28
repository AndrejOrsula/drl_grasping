ARG PARENT_IMAGE=nvidia/cudagl
ARG PARENT_IMAGE_TAG=11.4.2-devel-ubuntu20.04
FROM ${PARENT_IMAGE}:${PARENT_IMAGE_TAG}

### Use bash by default
SHELL ["/bin/bash", "-c"]

### Set non-interactive installation
ARG DEBIAN_FRONTEND=noninteractive

### Select Python version
ARG PYTHON_VERSION=3

### Install essentials, toolchain, python, cudnn...
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    apt-utils \
    locales \
    locales-all \
    tzdata \
    software-properties-common \
    git \
    wget \
    gnupg \
    lsb-release \
    build-essential \
    make \
    cmake \
    g++ \
    autoconf \
    automake \
    clang \
    ninja-build \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-wheel \
    python${PYTHON_VERSION}-numpy \
    python${PYTHON_VERSION}-tk \
    python${PYTHON_VERSION}-pybind11 \
    libpython${PYTHON_VERSION}-dev \
    libopenmpi-dev \
    zlib1g-dev \
    libcudnn8-dev \
    nano && \
    rm -rf /var/lib/apt/lists/*

### Install ROS 2
ARG ROS2_DISTRO=rolling
ENV ROS2_DISTRO=${ROS2_DISTRO}
RUN wget --progress=dot:giga https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list >/dev/null && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    ros-${ROS2_DISTRO}-ros-base \
    python${PYTHON_VERSION}-colcon-common-extensions \
    python${PYTHON_VERSION}-vcstool \
    python${PYTHON_VERSION}-argcomplete \
    python${PYTHON_VERSION}-rosdep && \
    rosdep init && rosdep update && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    rm -rf /var/lib/apt/lists/*

### Install Ignition
ARG IGNITION_VERSION=fortress
ENV IGNITION_VERSION=${IGNITION_VERSION}
RUN wget --progress=dot:giga https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    ignition-${IGNITION_VERSION} && \
    rm -rf /var/lib/apt/lists/*

### Set working directory
ARG WS_DIR=/root
ENV WS_DIR=${WS_DIR}
ENV WS_SRC_DIR=${WS_DIR}/src
ENV WS_INSTALL_DIR=${WS_DIR}/install
ENV ASSETS_DIR=${WS_DIR}/assets

### Install Python requirements (Torch, SB3, ...)
COPY ./python_requirements.txt ${WS_DIR}/python_requirements.txt
RUN pip${PYTHON_VERSION} install --no-cache-dir --upgrade pip && \
    pip${PYTHON_VERSION} install --no-cache-dir -r ${WS_DIR}/python_requirements.txt

### Setup token for accessing private UNILU GitLab repositories
ARG UNILU_GITLAB_ACCESS_TOKEN
RUN git config --global url."https://oauth2:${UNILU_GITLAB_ACCESS_TOKEN}@gitlab.uni.lu".insteadOf "https://gitlab.uni.lu"

### Clone all colcon-enabled repositories (dependencies)
COPY ./drl_grasping.repos ${WS_SRC_DIR}/drl_grasping.repos
WORKDIR ${WS_SRC_DIR}
RUN vcs import < ${WS_SRC_DIR}/drl_grasping.repos

### Install ROS dependencies and build with colcon
WORKDIR ${WS_DIR}
RUN rosdep update && \
    apt-get update && \
    rosdep install -r --from-paths ${WS_SRC_DIR} -yi --rosdistro ${ROS2_DISTRO} && \
    rm -rf /var/lib/apt/lists/*
### Install ROS dependencies that cannot be installed via rosdep
### TODO: Remove manual install once rosdep works again for rolling on focal (or once upgraded to humble)
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    ros-${ROS2_DISTRO}-ament-cmake \
    ros-${ROS2_DISTRO}-angles \
    ros-${ROS2_DISTRO}-backward-ros \
    ros-${ROS2_DISTRO}-control-msgs \
    ros-${ROS2_DISTRO}-control-toolbox \
    ros-${ROS2_DISTRO}-cv-bridge \
    ros-${ROS2_DISTRO}-geometric-shapes \
    ros-${ROS2_DISTRO}-interactive-markers \
    ros-${ROS2_DISTRO}-moveit-msgs \
    ros-${ROS2_DISTRO}-moveit-resources-fanuc-description \
    ros-${ROS2_DISTRO}-moveit-resources-fanuc-moveit-config \
    ros-${ROS2_DISTRO}-moveit-resources-panda-moveit-config \
    ros-${ROS2_DISTRO}-ompl \
    ros-${ROS2_DISTRO}-pluginlib \
    ros-${ROS2_DISTRO}-rclcpp \
    ros-${ROS2_DISTRO}-rclcpp-action \
    ros-${ROS2_DISTRO}-realtime-tools \
    ros-${ROS2_DISTRO}-ros-testing \
    ros-${ROS2_DISTRO}-ros2param \
    ros-${ROS2_DISTRO}-ros2run \
    ros-${ROS2_DISTRO}-rosidl-default-runtime \
    ros-${ROS2_DISTRO}-ruckig \
    ros-${ROS2_DISTRO}-rviz2 \
    ros-${ROS2_DISTRO}-sensor-msgs \
    ros-${ROS2_DISTRO}-srdfdom \
    ros-${ROS2_DISTRO}-std-msgs \
    ros-${ROS2_DISTRO}-std-srvs \
    ros-${ROS2_DISTRO}-tf2-eigen \
    ros-${ROS2_DISTRO}-tf2-msgs \
    ros-${ROS2_DISTRO}-tf2-ros \
    ros-${ROS2_DISTRO}-tinyxml2-vendor \
    ros-${ROS2_DISTRO}-trajectory-msgs \
    ros-${ROS2_DISTRO}-urdf \
    ros-${ROS2_DISTRO}-warehouse-ros \
    ros-${ROS2_DISTRO}-xacro && \
    rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

### Build iDynTree
WORKDIR ${WS_SRC_DIR}
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    coinor-libipopt-dev \
    libxml2-dev \
    qml-module-qt-labs-folderlistmodel \
    qml-module-qt-labs-settings \
    qml-module-qtmultimedia \
    qml-module-qtquick-controls \
    qml-module-qtquick-dialogs \
    qml-module-qtquick-window2 \
    qml-module-qtquick2 \
    qtbase5-dev \
    qtdeclarative5-dev \
    qtmultimedia5-dev \
    swig \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/robotology/idyntree --depth 1 -b v4.4.0 && \
    touch ${WS_SRC_DIR}/idyntree/COLCON_IGNORE && \
    mkdir ${WS_SRC_DIR}/idyntree/build && \
    cd ${WS_SRC_DIR}/idyntree/build && \
    cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DIDYNTREE_USES_PYTHON=True \
    -DIDYNTREE_USES_IPOPT:BOOL=ON && \
    cmake --build . --target install

### Build Gym-Ignition
WORKDIR ${WS_SRC_DIR}
RUN git clone https://github.com/AndrejOrsula/gym-ignition.git --depth 1 -b drl_grasping && \
    touch ${WS_SRC_DIR}/gym-ignition/COLCON_IGNORE && \
    mkdir -p ${WS_SRC_DIR}/gym-ignition/build && \
    cd ${WS_SRC_DIR}/gym-ignition/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . && \
    cmake --build . --target install && \
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade pip && \
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade numpy && \
    pip${PYTHON_VERSION} install --no-cache-dir .. && \
    pip${PYTHON_VERSION} install --no-cache-dir ../scenario

### Build O-CNN
WORKDIR ${WS_SRC_DIR}
RUN git clone https://github.com/AndrejOrsula/O-CNN.git --depth 1 -b master && \
    touch ${WS_SRC_DIR}/O-CNN/COLCON_IGNORE && \
    cd ${WS_SRC_DIR}/O-CNN/pytorch && \
    python${PYTHON_VERSION} setup.py install --build_octree

### Configure default datasets
WORKDIR ${ASSETS_DIR}
ARG DISABLE_DEFAULT_DATASETS
COPY ./scripts/utils/ ${ASSETS_DIR}/scripts/utils
RUN if [[ -z "${DISABLE_DEFAULT_DATASETS}" ]] ; then \
    echo "Downloading default datasets..." && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://gitlab.uni.lu/spacer/phd/AndrejOrsula/assets/textures.git --depth 1 -b master && \
    git clone https://gitlab.uni.lu/spacer/phd/AndrejOrsula/assets/sdf_models.git --depth 1 -b master && \
    ${ASSETS_DIR}/scripts/utils/dataset/dataset_download_test.bash \
    ; else \
    echo "Default datasets are disabled. Downloading skipped." \
    ; fi

# Install Dreamer v2
WORKDIR ${WS_SRC_DIR}
RUN git clone https://github.com/danijar/dreamerv2.git --depth 1 && \
    touch ${WS_SRC_DIR}/dreamerv2/COLCON_IGNORE && \
    cd ${WS_SRC_DIR}/dreamerv2 && \
    pip${PYTHON_VERSION} install --no-cache-dir .

# Install FFmpeg to enable dreamerv2 creating GIFs
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Downgrade `tensorflow` else it does not work (dreamerv2)
# Downgrade `markupsafe` else it does not work
# TODO: Revent once fixed
RUN pip${PYTHON_VERSION} install --no-cache-dir tensorflow==2.8.0 markupsafe==2.0.1

# Copy over drl_grasping repository and build it
COPY ./ ${WS_SRC_DIR}/drl_grasping/
WORKDIR ${WS_DIR}
RUN rosdep update && \
    apt-get update && \
    rosdep install -r --from-paths ${WS_SRC_DIR} -yi --rosdistro ${ROS2_DISTRO} && \
    rm -rf /var/lib/apt/lists/* && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release


### Go to the workspace root
WORKDIR ${WS_DIR}

### Set entrypoint and default command
RUN ln -sr ${WS_SRC_DIR}/drl_grasping/.docker/entrypoint.bash ${WS_DIR}/entrypoint.bash
ENTRYPOINT ["/bin/bash", "-c", "source ${WS_DIR}/entrypoint.bash && ${@}", "-s"]
CMD ["/bin/bash"]
