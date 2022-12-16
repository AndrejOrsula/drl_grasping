ARG PARENT_IMAGE=nvidia/cudagl
ARG PARENT_IMAGE_TAG=11.4.2-devel-ubuntu20.04
FROM ${PARENT_IMAGE}:${PARENT_IMAGE_TAG}

### Use bash by default
SHELL ["/bin/bash", "-c"]

### Define relevant directories
ARG HOME=/root
ARG WS_DIR=${HOME}/ws
ENV HOME=${HOME}
ENV WS_DIR=${WS_DIR}
ENV WS_SRC_DIR=${WS_DIR}/src
ENV WS_BUILD_DIR=${WS_DIR}/build
ENV WS_INSTALL_DIR=${WS_DIR}/install
ENV WS_LOG_DIR=${WS_DIR}/log
ENV ASSETS_DIR=${WS_DIR}/assets

### Select Python version
ARG PYTHON_VERSION=3

### Install Python, toolchain, cudnn and other essentials
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -yq --no-install-recommends \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    clang \
    cmake \
    g++ \
    git \
    gnupg \
    libcudnn8-dev \
    libopenmpi-dev \
    libpython${PYTHON_VERSION}-dev \
    locales \
    locales-all \
    make \
    nano \
    ninja-build \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-numpy \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-pybind11 \
    python${PYTHON_VERSION}-tk \
    python${PYTHON_VERSION}-wheel \
    software-properties-common \
    tzdata \
    wget \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

### Install ROS 2
ARG ROS_DISTRO=galactic
ENV ROS_DISTRO=${ROS_DISTRO}
RUN wget --progress=dot:giga https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo "${UBUNTU_CODENAME}") main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    python${PYTHON_VERSION}-argcomplete \
    python${PYTHON_VERSION}-colcon-common-extensions \
    python${PYTHON_VERSION}-rosdep \
    python${PYTHON_VERSION}-vcstool \
    ros-${ROS_DISTRO}-ros-base && \
    rosdep init && rosdep update && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rm -rf /var/lib/apt/lists/*

### Install Gazebo
ARG IGNITION_VERSION=fortress
ENV IGNITION_VERSION=${IGNITION_VERSION}
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    ignition-${IGNITION_VERSION} && \
    rm -rf /var/lib/apt/lists/*
## UNTIL FIXED: Compile ign-gazebo 6.9.0 from source because newer versions introduce errors
## Commit: https://github.com/gazebosim/gz-sim/commit/2938ede79feeb0fb1638370b910f06fb530be0ee
RUN git clone https://github.com/gazebosim/gz-sim.git -b ign-gazebo6 ${WS_SRC_DIR}/ign-gazebo && \
    git -C ${WS_SRC_DIR}/ign-gazebo reset --hard 2938ede79feeb0fb1638370b910f06fb530be0ee && \
    rm -rf ${WS_SRC_DIR}/ign-gazebo/.git
WORKDIR ${WS_DIR}
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release" && \
    rm -rf ${WS_LOG_DIR}

### Install Python requirements (Torch, SB3, ...)
COPY ./python_requirements.txt ${WS_SRC_DIR}/drl_grasping/python_requirements.txt
RUN pip${PYTHON_VERSION} install --no-cache-dir -r ${WS_SRC_DIR}/drl_grasping/python_requirements.txt

### Setup token for accessing private UNILU GitLab repositories
ARG UNILU_GITLAB_ACCESS_TOKEN
RUN git config --global url."https://oauth2:${UNILU_GITLAB_ACCESS_TOKEN}@gitlab.uni.lu".insteadOf "https://gitlab.uni.lu"

### Import and install dependencies, then build these dependencies (not drl_grasping yet)
WORKDIR ${WS_DIR}
COPY ./drl_grasping.repos ${WS_SRC_DIR}/drl_grasping/drl_grasping.repos
RUN vcs import --shallow ${WS_SRC_DIR} < ${WS_SRC_DIR}/drl_grasping/drl_grasping.repos && \
    rosdep update && \
    apt-get update && \
    rosdep install -y -r -i --rosdistro ${ROS_DISTRO} --from-paths ${WS_SRC_DIR} && \
    rm -rf /var/lib/apt/lists/* && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release" && \
    rm -rf ${WS_LOG_DIR}

### Install dependencies of drl_grasping
COPY ./package.xml ${WS_SRC_DIR}/drl_grasping/package.xml
RUN rosdep update && \
    apt-get update && \
    rosdep install -y -r -i --rosdistro ${ROS_DISTRO} --from-paths ${WS_SRC_DIR} && \
    rm -rf /var/lib/apt/lists/*

### Build iDynTree
WORKDIR ${WS_SRC_DIR}
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    coinor-libipopt-dev \
    libeigen3-dev \
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
    swig && \
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
    pip${PYTHON_VERSION} install --no-cache-dir numpy==1.21.6 && \
    pip${PYTHON_VERSION} install --no-cache-dir ${WS_SRC_DIR}/gym-ignition && \
    pip${PYTHON_VERSION} install --no-cache-dir -e ${WS_SRC_DIR}/gym-ignition/scenario

### Build O-CNN
WORKDIR ${WS_SRC_DIR}
RUN git clone https://github.com/AndrejOrsula/O-CNN.git --depth 1 -b master && \
    touch ${WS_SRC_DIR}/O-CNN/COLCON_IGNORE && \
    cd ${WS_SRC_DIR}/O-CNN/pytorch && \
    python${PYTHON_VERSION} ${WS_SRC_DIR}/O-CNN/pytorch/setup.py install --build_octree

### Install Dreamer v2
WORKDIR ${WS_SRC_DIR}
ARG INSTALL_DREAMERV2
RUN if [[ -n "${INSTALL_DREAMERV2}" ]] ; then \
    echo "Installing Dreamer V2..." && \
    git clone https://github.com/danijar/dreamerv2.git --depth 1 -b main && \
    touch ${WS_SRC_DIR}/dreamerv2/COLCON_IGNORE && \
    pip${PYTHON_VERSION} install --no-cache-dir ${WS_SRC_DIR}/dreamerv2 && \
    pip${PYTHON_VERSION} install --no-cache-dir \
    markupsafe==2.0.1 \
    protobuf==3.20.1 \
    tensorflow==2.8.0 \
    tensorflow-probability==0.16.0 && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/* \
    ; else \
    echo "Dreamer V2 is disabled. Installation skipped." \
    ; fi

### Configure default datasets
WORKDIR ${ASSETS_DIR}
COPY ./scripts/utils/dataset ${WS_SRC_DIR}/drl_grasping/scripts/utils/dataset
ARG DOWNLOAD_DATASETS
RUN if [[ -n "${DOWNLOAD_DATASETS}" ]] ; then \
    echo "Downloading default datasets..." && \
    ${WS_SRC_DIR}/drl_grasping/scripts/utils/dataset/dataset_download_test.bash && \
    ${WS_SRC_DIR}/drl_grasping/scripts/utils/dataset/dataset_download_train.bash && \
    ${WS_SRC_DIR}/drl_grasping/scripts/utils/dataset/dataset_set_train.bash && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    if [[ -n "${UNILU_GITLAB_ACCESS_TOKEN}" ]] ; then \
    echo "Downloading default textures and SDF models..." && \
    git clone https://gitlab.uni.lu/spacer/phd/AndrejOrsula/assets/textures.git --depth 1 -b master && \
    git clone https://gitlab.uni.lu/spacer/phd/AndrejOrsula/assets/sdf_models.git --depth 1 -b master \
    ; fi \
    ; else \
    echo "Default datasets are disabled. Downloading skipped." \
    ; fi

### Copy over the rest of drl_grasping, then build
WORKDIR ${WS_DIR}
COPY ./ ${WS_SRC_DIR}/drl_grasping/
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release" && \
    rm -rf ${WS_LOG_DIR}

### Setup symbolic links to simplify usage
### Source ROS workspace inside `~/.bashrc` to enable autocompletion
RUN "${WS_SRC_DIR}/drl_grasping/.docker/internal/setup_symlinks.bash" && \
    sed -i '$a source "/opt/ros/${ROS_DISTRO}/setup.bash"' ~/.bashrc

### Setup entrypoint
WORKDIR ${HOME}
ENTRYPOINT ["/bin/bash", "-c", "${HOME}/entrypoint.bash \"${@}\"", "-s"]
CMD ["/bin/bash"]
