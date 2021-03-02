from stable_baselines3.common.policies import register_policy, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.policies import Actor
from torch import nn
from typing import Any, Dict, List, Optional, Type, Union
import gym
import ocnn
import torch
import torch as th


class OctreeConvRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
        super(OctreeConvRelu, self).__init__()
        self.conv = ocnn.OctreeConv(depth,
                                    channel_in,
                                    channel_out,
                                    kernel_size,
                                    stride)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.relu(out)
        return out


class OctreeConvFastRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
        super(OctreeConvFastRelu, self).__init__()
        self.conv = ocnn.OctreeConvFast(depth,
                                        channel_in,
                                        channel_out,
                                        kernel_size,
                                        stride)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.relu(out)
        return out


class OctreeCnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param depth: Depth of input octree.
    :param full_depth: Depth at which convolutions stop and the octree is turned into voxel grid and flattened into output feature vector.
    :param channels_in: Number of input channels.
    :param channel_multiplier: Multiplier for the number of channels after each pooling. 
                               With this parameter set to 1, the channels are [1, 2, 4, 8, ...] for [depth, depth-1, ..., full_depth].
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 depth: int = 5,
                 full_depth: int = 2,
                 channels_in: int = 3,
                 channel_multiplier: int = 4):

        self._depth = depth
        self._channels_in = channels_in

        # Channels ordered as [channels_in, depth, depth-1, ..., full_depth]
        # I.e [channels_in, channel_multiplier*1, channel_multiplier*2, channel_multiplier*4, channel_multiplier*8,...]
        channels = [channel_multiplier*(2**i) for i in range(depth-full_depth)]
        channels.insert(0, channels_in)

        # The number of extracted features equals the channels at full_depth, times the number of cells in its voxel grid (O=3n)
        features_dim = channels[-1]*(2 ** (3 * full_depth))

        # Chain up parent constructor now that the dimension of the extracted features is known
        super(OctreeCnnFeaturesExtractor, self).__init__(observation_space,
                                                         features_dim)

        # Create all Octree convolution and pooling layers in depth-descending order [depth, depth-1, ..., full_depth]
        # Input to the first conv layer is the input Octree at depth=depth
        # Output from the last pool layer is feature map at depth=full_depth
        OctreeConv, OctreePool = OctreeConvFastRelu, ocnn.OctreeMaxPool
        self.convs = torch.nn.ModuleList([OctreeConv(depth-i, channels[i], channels[i+1])
                                          for i in range(depth-full_depth)])
        self.pools = torch.nn.ModuleList([OctreePool(depth-i)
                                          for i in range(depth-full_depth)])

        # Layer that converts octree at depth=full_depth into a full voxel grid (zero padding) such that it has a fixed size
        self.octree2voxel = ocnn.FullOctree2Voxel(full_depth)
        # Layer that flattens the voxel grid into a feature vector, this is the last layer of feature extractor that should feed into FC layers
        self.flatten = torch.nn.Flatten()

    def forward(self, octree):

        # Create a true Octree batch from the input
        # TODO: Use custom replay buffer thet creates batch properly. This would avoid 'CPU -> CUDA ->' CPU -> CUDA transfer
        octree_batch = ocnn.octree_batch(torch.split(octree.cpu(), 1)).cuda()

        # Extract features from the octree at the finest depth
        data = ocnn.octree_property(octree_batch, 'feature', self._depth)

        # Make sure the number of input channels matches the argument passed to constructor
        assert data.size(1) == self._channels_in

        # Pass the data through all convolutional and polling layers
        for i in range(len(self.convs)):
            data = self.convs[i](data, octree_batch)
            data = self.pools[i](data, octree_batch)

        # Convert octree at full_depth into a voxel grid
        data = self.octree2voxel(data)

        # Flatten into a feature vector
        data = self.flatten(data)

        return data


class ActorOctreeCnn(Actor):
    """
    Actor network (policy) for SAC.
    Overriden to not preprocess observations (unnecessary conversion into float)

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(ActorOctreeCnn, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
        Overriden to skip pre-processing (for some reason it converts tensor to Float)

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        return self.features_extractor(obs)
        # OVERRIDDEN
        # preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        # return self.features_extractor(preprocessed_obs)


class ContinuousCriticOctreeCnn(ContinuousCritic):
    """
    Critic network(s) for DDPG/SAC/TD3.
    Overriden to not preprocess observations (unnecessary conversion into float)

    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
        Overriden to skip pre-processing (for some reason it converts tensor to Float)

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        return self.features_extractor(obs)
        # OVERRIDDEN
        # preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        # return self.features_extractor(preprocessed_obs)


class OctreeCnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.
    Overriden to not preprocess observations (unnecessary conversion into float)

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use (``OctreeCnnFeaturesExtractor``).
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        # lr_schedule: Schedule, # Note: removed because hinting of Shedule results in import error
        lr_schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = OctreeCnnFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(OctreeCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor)
        return ActorOctreeCnn(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor)
        return ContinuousCriticOctreeCnn(**critic_kwargs).to(self.device)


register_policy("OctreeCnnPolicy", OctreeCnnPolicy)
