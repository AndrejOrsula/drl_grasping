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

### Setup token for accessing private UNILU GitLab repositories
ARG UNILU_GITLAB_ACCESS_TOKEN
RUN git config --global url."https://oauth2:${UNILU_GITLAB_ACCESS_TOKEN}@gitlab.uni.lu".insteadOf "https://gitlab.uni.lu"

### Clone drl_grasping and all colcon-enabled dependencies, then install Python requirements (Torch, SB3, ...)
WORKDIR ${WS_SRC_DIR}
ARG DRL_GRASPING_GIT_BRANCH=devel
# TODO: Add `--recursive` arg once pretrained agents are ready to use
RUN git clone https://gitlab.uni.lu/spacer/phd/AndrejOrsula/drl_grasping.git -b ${DRL_GRASPING_GIT_BRANCH} && \
    vcs import < ${WS_SRC_DIR}/drl_grasping/drl_grasping.repos && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${WS_SRC_DIR}/drl_grasping/python_requirements.txt

### Install ROS dependencies and build with colcon
WORKDIR ${WS_DIR}
RUN rosdep update && \
    apt-get update && \
    rosdep install -r --from-paths ${WS_SRC_DIR} -yi --rosdistro ${ROS2_DISTRO} && \
    rm -rf /var/lib/apt/lists/* && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
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
    git clone https://github.com/robotology/idyntree --depth 1 -b v4.3.1 && \
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
    pip install --no-cache-dir -e ..

### Build O-CNN
WORKDIR ${WS_SRC_DIR}
RUN git clone https://github.com/AndrejOrsula/O-CNN.git --depth 1 -b master && \
    touch ${WS_SRC_DIR}/O-CNN/COLCON_IGNORE && \
    cd ${WS_SRC_DIR}/O-CNN/pytorch && \
    python${PYTHON_VERSION} setup.py install --build_octree

### Configure default datasets
WORKDIR ${ASSETS_DIR}
ARG DISABLE_DEFAULT_DATASETS
RUN if [[ -z "${DISABLE_DEFAULT_DATASETS}" ]] ; then \
    echo "Downloading default datasets..." && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://gitlab.uni.lu/spacer/phd/AndrejOrsula/ign_fuel_assets.git --depth 1 -b master && \
    ./ign_fuel_assets/setup.bash \
    # && \
    # git clone https://github.com/AndrejOrsula/pbr_textures.git --depth 1 -b 1k_test && \
    # source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    # source ${WS_INSTALL_DIR}/local_setup.bash && \
    # export PATH=${WS_DIR}/O-CNN/octree/build:${PATH} && \
    # export PYTHONPATH=${WS_DIR}/O-CNN/octree/build/python:${PYTHONPATH} && \
    # ros2 run drl_grasping dataset_download_test.bash && \
    # ros2 run drl_grasping process_collection.py \
    ; else \
    echo "Default datasets are disabled. Downloading skipped." \
    ; fi

### Go to the workspace root
WORKDIR ${WS_DIR}

### Communicate within localhost only (Ignition)
ENV IGN_RELAY=127.0.0.1
ENV IGN_IP=127.0.0.1
### Communicate within localhost only (ROS 2)
ENV ROS_LOCALHOST_ONLY=1
### Set domain ID for ROS2 in order to not interfere with host
ENV ROS_DOMAIN_ID=69

### Set entrypoint and default command
COPY ./entrypoint.bash ${WS_DIR}/entrypoint.bash
ENTRYPOINT ["/bin/bash", "-c", "source ${WS_DIR}/entrypoint.bash && ${@}", "-s"]
CMD ["/bin/bash"]
