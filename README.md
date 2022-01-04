# Deep Reinforcement Learning for Robotic Grasping from Octrees

The focus of this project is to apply Deep Reinforcement Learning to acquire a robust policy that allows robots to grasp diverse objects from compact 3D observations in form of octrees.

Below are some animations of employing learned policies on novel scenes for Panda and UR5 robots.

<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim_panda.webp" alt="Evaluation of a trained policy on novel scenes for Panda robot"/>
</p>
<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim_ur5_rg2.webp" alt="Evaluation of a trained policy on novel scenes for UR5 robot"/>
</p>

Example of Sim2Real transfer on UR5 can be seen below (trained inside simulation, no re-training in real world).

<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim2real.webp" alt="Sim2Real evaluation of a trained policy on a real UR5 robot"/>
</p>

## Instructions

<details><summary>Local Installation (click to expand)</summary>

> If you just want to try this project without lengthy installation, consider using Docker instead.

### Requirements

- **OS:** Ubuntu 20.04 (Focal)
  - Others might work, but they were not tested.
- **GPU:** CUDA is required to process octree observations on GPU.
  - Everything else should function normally on CPU, i.e. environments with other observation types.

### Dependencies

These are the primary dependencies required to use this project.

- [Python 3.8](https://www.python.org/downloads)
- ROS 2 [Rolling](https://docs.ros.org/en/rolling/Installation.html)
  - [Foxy](https://docs.ros.org/en/foxy/Installation.html) and [Galactic](https://docs.ros.org/en/galactic/Installation.html) should also work without any issues (not tested)
- Ignition [Fortress](https://ignitionrobotics.org/docs/fortress)
  - [Citadel](https://ignitionrobotics.org/docs/citadel) and [Edifice](https://ignitionrobotics.org/docs/edifice) should also work (not tested)
- [MoveIt 2](https://moveit.ros.org/install-moveit2/binary)
  - Install/build a version based on the selected ROS 2 release
- [ros_ign](https://github.com/ignitionrobotics/ros_ign/tree/ros2)
  - Install/build a version based on the selected combination of ROS 2 release and Ignition version
- [ign_ros2_control](https://github.com/ignitionrobotics/ign_ros2_control)
  - Build a version based on the selected combination of ROS 2 release and Ignition version
- [gym-ignition](https://github.com/robotology/gym-ignition)
  - [AndrejOrsula/gym-ignition](https://github.com/AndrejOrsula/gym-ignition) fork is currently required
- [O-CNN](https://github.com/microsoft/O-CNN)
  - [AndrejOrsula/O-CNN](https://github.com/AndrejOrsula/O-CNN) fork is currently required
- [PyTorch](https://pytorch.org/get-started/locally) (last tested on 1.9.1)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) (last tested on 1.3.0) and [sb3-contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html#installation)

Additional Python dependencies are listed under [python_requirements.txt](./docker/python_requirements.txt). All of these (including Pytorch and Stable-Baselines3) can be installed via `pip`.

```bash
pip3 install -r python_requirements.txt
```

Additional dependencies for `pymoveit2` and robot models are listed under [drl_grasping.repos](./drl_grasping.repos) and pulled via git during installation. Please, see instructions below.

> In case you run into any problems with dependencies along the way, please check [Dockerfile](docker/Dockerfile) that includes the full instructions.

### Building

Clone this repository and import VCS dependencies. Then install dependencies and build with [colcon](https://colcon.readthedocs.io).

```bash
# Clone this repository into your favourite ROS 2 workspace
git clone https://github.com/AndrejOrsula/ign_moveit2_examples.git
# Import additional git dependencies
vcs import < ign_moveit2_examples/ign_moveit2_examples.repos
# Install external dependencies via rosdep
rosdep install -r --from-paths src --ignore-src --rosdistro ${ROS_DISTRO}
# Build with colcon
colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release"
```

> Use `git clone --recursive https://github.com/AndrejOrsula/drl_grasping.git` if you wish to use one of the pre-trained agents.

</details>

<details><summary>Docker (click to expand)</summary>

### Requirements

- **OS:** Any system that supports [Docker](https://docs.docker.com/get-docker) should work (Linux, Windows, macOS).
  - Only Ubuntu 20.04 was tested.
- **GPU:** CUDA is required to process octree observations on GPU. Therefore, only Docker images with CUDA support are currently available, however, it should be possible to use the pre-built image even on systems without a dedicated GPU.

### Dependencies

Before starting, make sure your system has a setup for using [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker), e.g.:

```bash
# Docker
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
# Nvidia Docker
distribution=$(. /etc/os-release; echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Pre-built Docker Image

The easiest way to try out this project is by using a pre-built Docker image that can be pulled from [Docker Hub](https://hub.docker.com/repository/docker/andrejorsula/drl_grasping). Currently, there is only a development image available that also contains the default testing datasets (huge, but it is easy to use and allows editing and recompiling). You can pull the `latest` tag with the following command. Alternatively, each release has also its associated tag, e.g. `1.0.0`.

```bash
docker pull andrejorsula/drl_grasping:latest
```

For running of the container, please use the included [docker/run.bash](docker/run.bash) script that is included with this repo. It significantly simplifies the setup with volumes and allows use of graphical interfaces for Ignition Gazebo GUI client and RViZ.

```bash
<drl_grasping dir>/docker/run.bash andrejorsula/drl_grasping:latest /bin/bash
```

If desired, you can also run examples and scripts directly with this setup, e.g. evaluation of pre-trained agents discussed below.

```bash
<drl_grasping dir>/docker/run.bash andrejorsula/drl_grasping:latest ros2 run drl_grasping ex_evaluate_pretrained_agent.bash
```

> If you are struggling to get CUDA working on your system with Nvidia GPU (no `nvidia-smi` output), you might need to use a different version of CUDA base image that supports the version of your driver. If that is the case, you need to build yourself a new Docker image.

### Building a New Image

[Dockerfile](docker/Dockerfile) is included with this repo but all source code is pulled from GitHub when building an image. There is nothing special about it, so just build it as any other Dockerfile (`docker build . -t ...`) and adjust arguments or the recipe itself if needed.

</details>

<details><summary>Sourcing of the Workspace Overlay (click to expand)</summary>

### Sourcing

Before running any commands, remember to source the ROS 2 workspace overlay. You can skip this step for Docker build as it is done automatically inside the entrypoint.

```bash
source <drl_grasping dir>/install/local_setup.bash
```

This enables:

- Use of `drl_grasping` Python module
- Execution of scripts and examples via `ros2 run drl_grasping <executable>`
- Launching of setup scripts via `ros2 launch drl_grasping <launch_script>`
- Discoverability of shared resources

</details>

<details><summary>Using Pre-trained Agents (click to expand)</summary>

### Evaluate Pre-trained Agents

The [pretrained_agents](https://github.com/AndrejOrsula/drl_grasping_pretrained_agents) submodule contains a selection of few agents that are already trained and ready to be evaluated (remember to `git clone --recursive`/`git submodule update --init` if you wish to use these). To use them, you can use [`ex_evaluate_pretrained_agent.bash`](examples/ex_evaluate_pretrained_agent.bash). You should see RViZ 2 and Ignition Gazebo GUI client with an agent trying to grasp one of four objects in a fully randomised novel environment, while the performance of the agent is logged in your terminal.

```bash
ros2 run drl_grasping ex_evaluate_pretrained_agent.bash
```

The default agent is for `Grasp-OctreeWithColor-Gazebo-v0` environment with Panda robot and TQC. You can modify these to any of the other pre-trained agent directly in the example script according to the support matrix from [AndrejOrsula/drl_grasping_pretrained_agents](https://github.com/AndrejOrsula/drl_grasping_pretrained_agents).

> Under the hood, all examples launch a setup ROS 2 script for interfacing Ignition, MoveIt 2 and a corresponding Python script for training or evaluation. All examples print these commands out if you are interested in running the commands separately.

</details>

<details><summary>Training New Agents (click to expand)</summary>

### Training of Agent

To train your own agent, you can start with the [`ex_train.bash`](examples/ex_train.bash) example. You can customise this example script,  configuration of the environment and all hyperparameters to your needs (see below). By default, headless mode is used during training to reduce computational load. If you want to see what is going on, use `ign gazebo -g` or `ROS_DOMAIN_ID=69 rviz2` (`ROS_DOMAIN_ID=69` is default for Docker image).

```bash
ros2 run drl_grasping ex_train.bash
```

Depending on your hardware and hyperparameter configuration, the training can be a very lengthy process. It takes nearly three days to train an agent for 500k steps on a 130W laptop with a dedicated GPU.

### Evaluation of Trained Agents

To evaluate an agent that you have trained yourself, look into [`ex_evaluate.bash`](examples/ex_evaluate.bash) example. Similar to training, change the environment ID, algorithm and robot model. Furthermore, select a specific checkpoint that you want to run. RViZ 2 and Ignition Gazebo GUI client are enabled by default.

```bash
ros2 run drl_grasping ex_evaluate.bash
```

</details>

## Environments

This repository contains environments for robotic manipulation that are compatible with [OpenAI Gym](https://github.com/openai/gym). All of these make use of [Ignition Gazebo](https://ignitionrobotics.org) robotic simulator, which is interfaced via [Gym-Ignition](https://github.com/robotology/gym-ignition).

Currently, the following environments are included inside this repository. Take a look at their [gym environment registration](drl_grasping/envs/tasks/__init__.py) and source code if you are interested in configuring them. There is a lot of parameters trying different RL approaches and techniques, so it is currently a bit messy (might get cleaned up if I have some free time for it).

- [Grasp](drl_grasping/envs/tasks/grasp) task (the focus of this project)
  - Observation variants
    - [GraspOctree](drl_grasping/envs/tasks/grasp/grasp_octree.py), with and without color features
    - GraspColorImage (RGB image) and GraspRgbdImage (RGB-D image) are implemented on [image_obs](https://github.com/AndrejOrsula/drl_grasping/tree/image_obs) branch. However, their implementation is currently only for testing and comparative purposes.
  - Curriculum Learning: Task includes [GraspCurriculum](drl_grasping/envs/tasks/grasp/curriculum.py), which can be used to progressively increase difficulty of the task by automatically adjusting the following environment parameters based on the current success rate.
    - Workspace size
    - Number of objects
    - Termination state (task is divided into hierarchical sub-tasks with aim to further guide the agent).
      - This part does not bring any improvements based on experimental results, so do not bother using it.
  - Demonstrations: Task contains a simple scripted policy that can be applied to collect demonstrations, which can then be used to pre-load a replay buffer for training with off-policy RL algorithms.
    - It provides a slight increase for early learning, however, experiments indicate that it degrades the final success rate (probably due to introduction of bias early on). Therefore, do not use demonstrations if possible, at least not with this environment.
- [Reach](drl_grasping/envs/tasks/reach) task (a simplistic environment for testing stuff)
  - Observation variants
    - [Reach](drl_grasping/envs/tasks/reach/reach.py) - simulation states
    - [ReachColorImage](drl_grasping/envs/tasks/reach/reach_color_image.py)
    - [ReachDepthImage](drl_grasping/envs/tasks/reach/reach_depth_image.py)
    - [ReachOctree](drl_grasping/envs/tasks/reach/reach_octree.py), with and without color features

### Domain Randomization

These environments can be wrapped by a randomizer in order to introduce domain randomization and improve generalization of the trained policies, which is especially beneficial for Sim2Real transfer.

<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/graphics/implementation/domain_randomisation.png" alt="Examples of domain randomization for the Grasp task"/>
</p>

The included [ManipulationGazeboEnvRandomizer](drl_grasping/envs/randomizers/manipulation.py) allows randomization of the following properties at each reset of the environment.

- Object model - primitive geometry
  - Random type (box, sphere and cylinder are currently supported)
  - Random color, scale, mass, friction
- Object model - mesh geometry
  - Random type (see [Dataset of Object Models](#dataset-of-object-models))
  - Random scale, mass, friction
- Object pose
- Ground plane texture
- Initial robot configuration
- Camera pose

#### Dataset of Object Models

For dataset of objects with mesh geometry and material texture, this project utilizes [Google Scanned Objects collection](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects) from [Ignition Fuel](https://app.ignitionrobotics.org). You can also try to use a different Fuel collection or just a couple of models stored locally (although some tweaks might be required to support certain models).

All models are automatically configured in several ways before their insertion into the world:

- Inertial properties are automatically estimated (uniform density is assumed)
- Collision geometry is decimated in order to improve performance
- Models can be filtered and automatically blacklisted based on several aspects, e.g too much geometry or disconnected components

This repository includes few scripts that can be used to simplify interaction with the dataset and splitting into training/testing subsets. By default they include 80 training and 20 testing models.

- [`dataset_download_train`](scripts/utils/dataset/dataset_download_train.bash) / [`dataset_download_test`](scripts/utils/dataset/dataset_download_test.bash) - Download models from Fuel
- [`dataset_unset_train`](scripts/utils/dataset/dataset_unset_train.bash) / [`dataset_unset_test`](scripts/utils/dataset/dataset_unset_test.bash) - Unset current train/test dataset
- [`dataset_set_train`](scripts/utils/dataset/dataset_set_train.bash) / [`dataset_set_test`](scripts/utils/dataset/dataset_set_test.bash) - Set dataset to use train/test subset
- [`process_collection`](scripts/utils/process_collection.py) - Process the collection with the steps mentioned above

#### Texture Dataset

`DRL_GRASPING_PBR_TEXTURES_DIR` environment variable can be exported if ground plane texture should be randomized. It should lead to a directory with the following structure.

```bash
├── ./ # Directory pointed to by `DRL_GRASPING_PBR_TEXTURES_DIR`
├── texture_0
  ├── *albedo*.png || *basecolor*.png
  ├── *normal*.png
  ├── *roughness*.png
  └── *specular*.png || *metalness*.png
├── ...
└── texture_n
```

There are several databases with free PBR textures that you can use. Alternatively, you can clone [AndrejOrsula/pbr_textures](https://github.com/AndrejOrsula/pbr_textures) with 80 training and 20 testing textures.

### Supported Robots

Only [Franka Emika Panda](https://github.com/AndrejOrsula/panda_ign), [UR5 with RG2 gripper](https://github.com/AndrejOrsula/ur5_rg2_ign) and [Kinova Gen2 (j2s7s300)](https://github.com/AndrejOrsula/kinova_j2s7s300_ign) are supported. This project currently lacks a more generic solution that would allow to easily utilize arbitrary models, e.g. full-on [MoveIt 2](https://github.com/ros-planning/moveit2) with [ros2_control](https://github.com/ros-controls/ros2_control) implementation. Adding new models is not complicated though, just time-consuming.

## Reinforcement Learning

This project makes direct use of [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) as well as [sb3_contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib). Furthermore, scripts for training and evaluation are largely inspired by [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

### Octree CNN Features Extractor

The [OctreeCnnFeaturesExtractor](drl_grasping/drl_octree/features_extractor/octree_cnn.py) makes use of [O-CNN](https://github.com/microsoft/O-CNN) implementation to enable training on GPU. This features extractor is part of `OctreeCnnPolicy` policy that is currently implemented for TD3, SAC and TQC algorithms. Network architecture of this feature extractor is illustrated below.

<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/svg/feature_extractor.svg" alt="Architecture of octree-based 3D CNN feature extractor"/>
</p>

### Hyperparameters

Hyperparameters for training of RL agents can be found in [hyperparams](hyperparams) directory. [Optuna](https://github.com/optuna/optuna) was used to autotune some of them, but certain algorithm/environment combinations require far more tuning (especially TD3). If needed, you can try running Optuna yourself, see [`ex_optimize`](examples/ex_optimize.bash) example.

## Directory Structure

```bash
├── drl_grasping        # Primary Python module of this project
    ├── algorithms      # Definitions of policies and slight modifications to RL algorithms
    ├── envs            # Environments for grasping (compatible with OpenAI Gym)
        ├── tasks       # Tasks for the agent that are identical for simulation
        ├── randomizers # Domain randomization of the tasks, which also populates the world
        └── models      # Functional models for the environment (Ignition Gazebo)
    ├── control         # Control for the agent
    ├── perception      # Perception for the agent
    └── utils           # Other utilities, used across the module
├── examples            # Examples for training and evaluating RL agents
├── hyperparams         # Hyperparameters for training RL agents
├── scripts             # Helpful scripts for training, evaluating, ...
├── launch              # ROS 2 launch scripts that can be used to help with setup
├── docker              # Dockerfile for this project
└── drl_grasping.repos  # List of other dependencies created for `drl_grasping`
```

______________________________________________________________________

In case you have any problems or questions, feel free to open an [Issue](https://github.com/AndrejOrsula/drl_grasping/issues/new) or a [Discussion](https://github.com/AndrejOrsula/drl_grasping/discussions/new).
