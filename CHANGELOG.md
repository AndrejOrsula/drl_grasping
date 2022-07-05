# CHANGELOG

## [\[2.0.0\] - 2022-07-18](https://github.com/AndrejOrsula/drl_grasping/releases/tag/2.0.0)

### Added

- Full integration with `ros2_control`.
- Support for `moveit_servo` via `pymoveit`.
- Support for mobile manipulators.
- `lunalab_summit_xl_gen`  is now added to the supported (mobile) robot models.
- Configuration of pre-commit git hooks.
- Models for `Sun` and `RandomSun`, replacing the default light contained in SDF worlds.
- Camera can now be mounted relative to robot frame (configurable) via DetachableJoint.
- Added experimental setup for Dreamer V2 algorithm.

### Changed

- Major refactoring of `drl_grasping` module
  - Refactored into two primary submodules that can be imported separately
    - `drl_octree` that contains octree CNN-based policy
    - `envs` that contain the environment itself
  - `utils` submodule still contains boilerplate for RL training and evaluation
- Randomizer is now used to wrap task environment during gym environment registration.
  - This means, that two environment variants of each task now exist, i.e. `*-v0` and `*-Gazebo-v0`. The default task can therefore be used with different run-times without requiring changes to the hyperparameters in order to make it functional.
- Environments are now registered directly in `envs` module instead of `envs.tasks`
- All important features are now exposed as ROS 2 launch scripts.
- A single configurable launch script [sim.launch.py](./launch/sim.launch.py) now replaces all previous variations. Use its launch arguments to select robot model and enable/disable RViz2 GUI.
- Instead of all fingers, more than half (`n//2 + 1`) are now needed to be in contact for a grasp to be successful
- Changed PEP 8 python formatting for black to improve consistency.
- Changed bash formatter to beautysh (minor changes).
- Custom environment logging is now performed via ROS 2 loggers.
- Local classes for interfacing with MoveIt 2.
- Simplified usage and quality of life improvements.

### Fixed

- Fix grasp checking for grippers with more than 2 fingers.

### Removed

- `DRL_GRASPING_DEBUG_LEVEL` environment variable is now replaced by `log-level` ROS 2 argument.
- `ur5_rg2` and `kinova_j2s7s300` from supported robot models due to shift to `ros2_control`.

## [\[1.1.0\] - 2021-10-13](https://github.com/AndrejOrsula/drl_grasping/releases/tag/1.1.0)

### Added

- `kinova_j2s7s300` is now added to the supported robot models.
- Custom SDF worlds are now used as the base for RL environments.
- Support for `DRL_GRASPING_DEBUG_LEVEL` environment variable.
- Ignition fortress is tested and fully functional.
- ROS 2 rolling is tested and fully functional.

### Changed

- Local implementation of conversions for quaternion sequence is now used.
- Simplified installation instructions in README.md.
- Simplified and improved Dockerfile.

### Fixed

- Compatibility with Stable-Baselines3 v1.2.0

## [\[1.0.0\] - 2021-06-08](https://github.com/AndrejOrsula/drl_grasping/releases/tag/1.0.0)

### Added

- Initial version of this project.
  - Supported environments
    - `Reach-Gazebo-v0`
    - `Reach-ColorImage-Gazebo-v0`
    - `Reach-Octree-Gazebo-v0`
    - `Reach-OctreeWithColor-Gazebo-v0`
    - `Grasp-Octree-Gazebo-v0`
    - `Grasp-OctreeWithColor-Gazebo-v0`
  - Supported robot models
    - `panda`
    - `ur5_rg2`
  - Custom feature extractor
    - `OctreeCnnFeaturesExtractor`
  - Tested RL algorithms
    - `td3`
    - `sac`
    - `tqc`
