# Deep Reinforcement Learning for Robotic Grasping on Point Clouds

**Note: This project is currently very much WIP, so please ignore it for your own good**

This is the primary repository for my Master's Thesis conducted at Aalborg University, Denmark. Deadline for project is Jun 2021. The focus of the project is on DRL agent with policy that allows grasping of arbitrary objects based on point cloud observations of scene that is perceived by RGB-D camera(s).

- TODO: Document project further once it takes its form

Note: There might be some modules/scripts in this repository that are useful also for other projects when working with Ignition Gazebo, e.g. [model_collection_randomizer.py](drl_grasping/utils/model_collection_randomizer.py). I might eventually move them out of this repo and develop independently, therefore, please treat their current presence as temporary.

## Directory Structure

```bash
├── drl_grasping          # Primary Python module of this project
    ├── algorithms        # DRL algorithms and definitions of policies (WIP)
    ├── envs              # Environments for grasping (based on gym-ignition, compatible with OpenAI Gym)
        ├── tasks         # Tasks for the agent that are identical for simulation and real-world
        ├── randomizers   # Domain randomization of the tasks, which also populates the simulated world (Ignition Gazebo)
        ├── models        # Functional models for the environment (Ignition Gazebo)
        └── worlds        # Worlds (Ignition Gazebo)
    ├── control           # Control for the agent
    ├── perception        # Perception for the agent
    └── utils             # Other utilities, used across the module
├── launch                # ROS2 launch scripts helping with setup
├── scripts               # Python scripts used to training and testing
└── drl_grasping.repos    # List of other dependencies created for `drl_grasping` (with git links)
```

## Instructions

**Note: These instructions are WIP and there is a large possibility that they are incomplete**

- TODO: Make sure installation instructions are correct

### Dependencies

- [ROS 2 Foxy](https://index.ros.org/doc/ros2/Installation/Foxy)
- [MoveIt2](https://moveit.ros.org/install-moveit2/source)
- [Ignition Dome](https://ignitionrobotics.org/docs/dome/install)
  - `ign-msgs` >= 6.2
- [ros_ign](https://github.com/ignitionrobotics/ros_ign/tree/ros2)
  - `ros2` branch with <https://github.com/ignitionrobotics/ros_ign/pull/121>

Other dependencies can be installed via `pip` (tested with `Python 3.8`).

```bash
pip install numpy
pip install scipy
pip install trimesh
pip install open3d
pip install pcg-gazebo
pip install torch
```

- TODO: Document `gym-ignition` installation instructions (currently using <https://github.com/AndrejOrsula/gym-ignition> fork).

All other dependencies are pulled from git ([ign_moveit2.repos](ign_moveit2.repos)) and built automatically with this repository.

### Building

<!-- Clone, clone dependencies and build with `colcon`.

```bash
export PARENT_DIR=${PWD}
mkdir -p drl_grasping/src && cd drl_grasping/src
git clone https://github.com/AndrejOrsula/drl_grasping.git -b master
vcs import < drl_grasping/drl_grasping.repos
cd ..
rosdep install --from-paths src -i -y --rosdistro ${ROS_DISTRO}
colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release"
``` -->

### Usage

#### Environment

Source the ROS 2 workspace overlay.

```bash
source ${PARENT_DIR}/drl_grasping/install/local_setup.bash
```

#### Random objects from Ignition Fuel

This project currently utilises [Google Scanned Objects collection](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects) from [Ignition Fuel](https://app.ignitionrobotics.org/) to serve as dataset of training objects. These models are automatically configured in several ways before their insertion into the world:

- Inertial properties are automatically estimated (assuming uniform density)
- Collision geometry is decimated (to improve performance)
- Models can be filtered and automatically blacklisted based on several aspects, e.g too much geometry or disconnected components

All models can also be randomized prior to each insertion in several aspect (inertial properties are automatically adjusted).

- Random scale
- Random mass
- Random friction

It is recommended (but optional) to run [process_all_models.py](scripts/utils/process_all_models.py) with your custom parameters to configure all the models. The entire collection will be automatically downloaded into your local cache during the first execution such that future runs need to only search locally, as getting list of models from Fuel can sometimes be slow for large collections.

You can try to use a different Fuel collection or just a couple of models stored locally. However, it might not function properly for certain models as it is only tested with the aforementioned collection.

Note: Needs to be enabled in randomizer

#### Ground plane with random PBR textures

To get ground plane with random PBR textures, `DRL_GRASPING_PBR_TEXTURES_DIR` environment variable can be exported. It should lead to directory with the following structure.

```bash
├── ./                                   # Directory pointed to by `DRL_GRASPING_PBR_TEXTURES_DIR`
├── texture_0                            # PBR texture with name
  ├── *albedo*.png || *basecolor*.png    # Search terms are case-insensitive
  ├── *normal*.png
  ├── *roughness*.png
  └── *specular*.png || *metalness*.png
├── texture_1
└── texture_n
```

There are several databases with free PBR textures, e.g <https://texturefun.com/>, however, I am in no position to redistribute them.

Note: Needs to be enabled in randomizer

<!-- #### Examples -->
