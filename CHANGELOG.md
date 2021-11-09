# CHANGELOG
  
## [Unreleased] - yyyy-mm-dd

### Added

- (WIP) `lunalab_summit_xl_gen` is now added to the supported robot models.

### Changed

- Instead of all fingers, more than half (`n//2 + 1`) are now needed to be in contact for a grasp to be successful
- Major refactoring of `drl_grasping` module
  - Refactored into two primary submodules that can be imported separately
    - `drl_octree` that contains octree CNN-based policy
    - `envs` that contain the environment itself
  - `utils` submodule still contains boilerplate for RL training and evaluation
- A single configurable launch script [sim.launch.py](./launch/sim.launch.py) now replaces all previous variations. Use its launch arguments to select robot model and enable/disable RViz2 GUI.

### Fixed
 
- Fix grasp checking for grippers with more than 2 fingers.

## [[1.1.0] - 2021-10-13](https://github.com/AndrejOrsula/drl_grasping/releases/tag/1.1.0)
 
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


## [[1.0.0] - 2021-06-08](https://github.com/AndrejOrsula/drl_grasping/releases/tag/1.0.0)
 
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
