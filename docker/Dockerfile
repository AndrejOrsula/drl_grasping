ARG PARENT_IMAGE=nvidia/cudagl:11.4.2-devel-ubuntu20.04
FROM ${PARENT_IMAGE}

### Use bash by default
SHELL ["/bin/bash", "-c"]

### Set non-interactive installation
ARG DEBIAN_FRONTEND=noninteractive

### Set working directory
ARG DRL_GRASPING_DIR=/root/drl_grasping
ENV DRL_GRASPING_DIR=${DRL_GRASPING_DIR}
WORKDIR ${DRL_GRASPING_DIR}

### Install essentials, toolchain, python, cudnn...
ARG PYTHON_VERSION=3
RUN apt update && \
    apt install -yq --no-install-recommends \
        apt-utils \
        locales \
        locales-all \
        tzdata \
        software-properties-common \
        git \
        wget \
        curl \
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
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list >/dev/null && \
    apt update && \
    apt install -yq --no-install-recommends \
        ros-${ROS2_DISTRO}-desktop \
        python${PYTHON_VERSION}-colcon-common-extensions \
        python${PYTHON_VERSION}-vcstool \
        python${PYTHON_VERSION}-argcomplete \
        python${PYTHON_VERSION}-rosdep && \
    rosdep init && rosdep update && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    rm -rf /var/lib/apt/lists/*

### Install MoveIt 2
RUN apt update && \
    apt install -yq --no-install-recommends \
        ros-${ROS2_DISTRO}-moveit-common \
        ros-${ROS2_DISTRO}-moveit-core \
        ros-${ROS2_DISTRO}-moveit-kinematics \
        ros-${ROS2_DISTRO}-moveit-msgs \
        ros-${ROS2_DISTRO}-moveit-planners \
        ros-${ROS2_DISTRO}-moveit-planners-ompl \
        ros-${ROS2_DISTRO}-moveit-plugins \
        ros-${ROS2_DISTRO}-moveit-resources \
        ros-${ROS2_DISTRO}-moveit-ros \
        ros-${ROS2_DISTRO}-moveit-ros-occupancy-map-monitor \
        ros-${ROS2_DISTRO}-moveit-ros-perception \
        ros-${ROS2_DISTRO}-moveit-ros-planning \
        ros-${ROS2_DISTRO}-moveit-ros-planning-interface \
        ros-${ROS2_DISTRO}-moveit-runtime \
        ros-${ROS2_DISTRO}-moveit-servo \
        ros-${ROS2_DISTRO}-moveit-simple-controller-manager && \
    rm -rf /var/lib/apt/lists/*

### Install Ignition
ARG IGNITION_VERSION=fortress
ENV IGNITION_VERSION=${IGNITION_VERSION}
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt update && \
    apt install -yq --no-install-recommends \
    ignition-${IGNITION_VERSION} && \
    rm -rf /var/lib/apt/lists/*

### Build ROS 2 <-> IGN
RUN mkdir -p ros_ign/src && \
    cd ros_ign && \
    git clone https://github.com/ignitionrobotics/ros_ign.git --depth 1 -b ros2 src && \
    apt update && \
    rosdep update && \
    rosdep install -r --from-paths . -yi --rosdistro ${ROS2_DISTRO} && \
    rm -rf /var/lib/apt/lists/* && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    source install/local_setup.bash
WORKDIR ${DRL_GRASPING_DIR}

### Install Python requirements (Torch, SB3, ...)
COPY ./python_requirements.txt ./python_requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r python_requirements.txt

### Build O-CNN
RUN git clone https://github.com/AndrejOrsula/O-CNN.git --depth 1 && \
    cd O-CNN/pytorch && \
    python${PYTHON_VERSION} setup.py install --build_octree && \
    cd ../octree/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON -DUSE_PYTHON=ON -DABI=ON -DKEY64=ON && \
    cmake --build . --config Release
WORKDIR ${DRL_GRASPING_DIR}

### Build iDynTree
RUN apt update && \
    apt install -yq --no-install-recommends \
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
    git clone https://github.com/robotology/idyntree --depth 1 && \
    mkdir idyntree/build && \
    cd idyntree/build && \
    cmake .. \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS:BOOL=OFF \
        -DIDYNTREE_USES_PYTHON=True \
        -DIDYNTREE_USES_IPOPT:BOOL=ON && \
    cmake --build . --target install
WORKDIR ${DRL_GRASPING_DIR}

### Build Gym-Ignition
RUN git clone https://github.com/AndrejOrsula/gym-ignition.git --depth 1 -b drl_grasping && \
    mkdir -p gym-ignition/build && \
    cd gym-ignition/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . && \
    cmake --build . --target install && \
    pip install .. && \
    pip install ../scenario
WORKDIR ${DRL_GRASPING_DIR}

### Main repository
ARG DRL_GRASPING_GIT_BRANCH=master
ARG DRL_GRASPING_DOCKER_VERSION=1.1.0
RUN mkdir -p drl_grasping/src && \
    cd drl_grasping/src && \
    git clone https://github.com/AndrejOrsula/drl_grasping.git --recursive --depth 1 -b ${DRL_GRASPING_GIT_BRANCH} && \
    vcs import < drl_grasping/drl_grasping.repos && \
    cd .. && \
    apt update && \
    rosdep update && \
    rosdep install -r --from-paths . -yi --rosdistro ${ROS2_DISTRO} && \
    rm -rf /var/lib/apt/lists/* && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    source install/local_setup.bash
WORKDIR ${DRL_GRASPING_DIR}

### Download and process default datasets (if desired)
ARG DISABLE_DEFAULT_DATASETS
RUN if [[ -z "${DISABLE_DEFAULT_DATASETS}" ]] ; then \
        echo "Downloading default datasets..." && \
        source /opt/ros/${ROS2_DISTRO}/setup.bash && \
        source ${DRL_GRASPING_DIR}/drl_grasping/install/local_setup.bash && \
        export PATH=${DRL_GRASPING_DIR}/O-CNN/octree/build:${PATH} && \
        export PYTHONPATH=${DRL_GRASPING_DIR}/O-CNN/octree/build/python:${PYTHONPATH} && \
        ign fuel download -t model -u https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/panda && \
        ign fuel download -t model -u https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/ur5_rg2 && \
        ign fuel download -t model -u https://fuel.ignitionrobotics.org/1.0/AndrejOrsula/models/kinova_j2s7s300 && \
        ros2 run drl_grasping dataset_download_test.bash && \
        ros2 run drl_grasping process_collection.py && \
        git clone https://github.com/AndrejOrsula/pbr_textures.git --depth 1 -b 1k_test default_pbr_textures \
    ; else \
        echo "Default datasets are disabled. Downloading skipped." \
    ; fi

### Communicate within localhost only
ENV ROS_LOCALHOST_ONLY=1
### Set domain ID for ROS2 in order to not interfere with host
ENV ROS_DOMAIN_ID=69

### Set debug level
ENV DRL_GRASPING_DEBUG_LEVEL=ERROR

### Add entrypoint sourcing the environment
COPY ./entrypoint.bash ./entrypoint.bash

### Set entrypoint and default command
ENTRYPOINT ["/bin/bash", "-c", "source ${DRL_GRASPING_DIR}/entrypoint.bash && \"$@\"", "-s"]
CMD ["/bin/bash"]
