import gym
import ocnn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from drl_grasping.drl_octree.features_extractor.modules import *

# TODO: Once it is clear whether separating networks for stacks or not makes a difference (e.g. better stability), refactor this mess and use only the better solution. Same applies to other args... there are too many of them right now


class OctreeCnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param depth: Depth of input octree.
    :param full_depth: Depth at which convolutions stop and the octree is turned into voxel grid and flattened into output feature vector.
    :param channels_in: Number of input channels.
    :param channel_multiplier: Multiplier for the number of channels after each pooling.
                               With this parameter set to 1, the channels are [1, 2, 4, 8, ...] for [depth, depth-1, ..., full_depth].
    :param features_dim: Dimension of output feature vector. Note that this number is multiplied by the number of stacked octrees inside one observation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        depth: int = 5,
        full_depth: int = 2,
        channels_in: int = 4,
        channel_multiplier: int = 16,
        full_depth_conv1d: bool = False,
        full_depth_channels: int = 8,
        features_dim: int = 128,
        aux_obs_dim: int = 0,
        aux_obs_features_dim: int = 10,
        separate_networks_for_stacks: bool = True,
        fast_conv: bool = True,
        batch_normalization: bool = True,
        bn_eps: float = 0.00001,
        bn_momentum: float = 0.01,
        verbose: bool = False,
    ):

        self._depth = depth
        self._channels_in = channels_in
        self._aux_obs_dim = aux_obs_dim
        self._aux_obs_features_dim = aux_obs_features_dim
        self._separate_networks_for_stacks = separate_networks_for_stacks
        self._verbose = verbose

        # Determine what type of octree-based convolutions to use
        if fast_conv:
            if batch_normalization:
                OctreeConv = OctreeConvFastBnRelu
                OctreeConv1D = OctreeConv1x1BnRelu
            else:
                OctreeConv = OctreeConvFastRelu
                OctreeConv1D = OctreeConv1x1Relu
        else:
            if batch_normalization:
                OctreeConv = OctreeConvBnRelu
                OctreeConv1D = OctreeConv1x1BnRelu
            else:
                OctreeConv = OctreeConvRelu
                OctreeConv1D = OctreeConv1x1Relu
        OctreePool = ocnn.OctreeMaxPool
        # Keyword arguments used for layers that might contain BatchNorm layers
        bn_kwargs = {}
        if batch_normalization:
            bn_kwargs.update({"bn_eps": bn_eps, "bn_momentum": bn_momentum})

        # Determine number of stacked octrees based on observation space shape
        self._n_stacks = observation_space.shape[0]

        # Chain up parent constructor
        super(OctreeCnnFeaturesExtractor, self).__init__(
            observation_space, self._n_stacks * (features_dim + aux_obs_features_dim)
        )

        # Channels ordered as [channels_in, depth, depth-1, ..., full_depth]
        # I.e [channels_in, channel_multiplier*1, channel_multiplier*2, channel_multiplier*4, channel_multiplier*8,...]
        self._n_convs = depth - full_depth
        channels = [channel_multiplier * (2 ** i) for i in range(self._n_convs)]
        channels.insert(0, channels_in)

        full_depth_voxel_count = 2 ** (3 * full_depth)
        flatten_dim = full_depth_channels * full_depth_voxel_count

        if not self._separate_networks_for_stacks:

            # Create all Octree convolution and pooling layers in depth-descending order [depth, depth-1, ..., full_depth]
            # Input to the first conv layer is the input Octree at depth=depth
            # Output from the last pool layer is feature map at depth=full_depth
            self.convs = torch.nn.ModuleList(
                [
                    OctreeConv(depth - i, channels[i], channels[i + 1], **bn_kwargs)
                    for i in range(self._n_convs)
                ]
            )
            self.pools = torch.nn.ModuleList(
                [OctreePool(depth - i) for i in range(self._n_convs)]
            )

            # Last convolution at depth=full_depth, which is not follewed by pooling
            # This layer is used to significantly reduce the channels, decresing number of parameters required in the next linear layer(s)
            self._full_depth_conv1d = full_depth_conv1d
            if self._full_depth_conv1d:
                # Use 1D convolution (Conv1D instead of linear is used here to preserve spatial information)
                self.full_depth_conv = OctreeConv1D(
                    channels[-1], full_depth_channels, **bn_kwargs
                )
            else:
                # Use 3D convolution (same as previous modules)
                self.full_depth_conv = OctreeConv(
                    full_depth, channels[-1], full_depth_channels, **bn_kwargs
                )

            # Layer that converts octree at depth=full_depth into a full voxel grid (zero padding) such that it has a fixed size
            self.octree2voxel = ocnn.FullOctree2Voxel(full_depth)

            # Layer that flattens the voxel grid into a feature vector
            self.flatten = torch.nn.Flatten()

            # Last linear layer of the extractor, applied to all (flattened) voxels at full depth
            self.linear = LinearRelu(flatten_dim, features_dim)

            # One linear layer for auxiliary observations
            if self._aux_obs_dim != 0:
                self.aux_obs_linear = LinearRelu(
                    self._aux_obs_dim, aux_obs_features_dim
                )

        else:

            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [
                            OctreeConv(
                                depth - i, channels[i], channels[i + 1], **bn_kwargs
                            )
                            for i in range(self._n_convs)
                        ]
                    )
                    for _ in range(self._n_stacks)
                ]
            )
            self.pools = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [OctreePool(depth - i) for i in range(self._n_convs)]
                    )
                    for _ in range(self._n_stacks)
                ]
            )

            # Last convolution at depth=full_depth, which is not follewed by pooling
            # This layer is used to significantly reduce the channels, decresing number of parameters required in the next linear layer(s)
            self._full_depth_conv1d = full_depth_conv1d
            if self._full_depth_conv1d:
                # Use 1D convolution (Conv1D instead of linear is used here to preserve spatial information)
                self.full_depth_conv = torch.nn.ModuleList(
                    [
                        OctreeConv1D(channels[-1], full_depth_channels, **bn_kwargs)
                        for _ in range(self._n_stacks)
                    ]
                )
            else:
                # Use 3D convolution (same as previous modules)
                self.full_depth_conv = torch.nn.ModuleList(
                    [
                        OctreeConv(
                            full_depth, channels[-1], full_depth_channels, **bn_kwargs
                        )
                        for _ in range(self._n_stacks)
                    ]
                )

            # Layer that converts octree at depth=full_depth into a full voxel grid (zero padding) such that it has a fixed size
            self.octree2voxel = torch.nn.ModuleList(
                [ocnn.FullOctree2Voxel(full_depth) for _ in range(self._n_stacks)]
            )

            # Layer that flattens the voxel grid into a feature vector
            self.flatten = torch.nn.ModuleList(
                [torch.nn.Flatten() for _ in range(self._n_stacks)]
            )

            # Last linear layer of the extractor, applied to all (flattened) voxels at full depth
            self.linear = torch.nn.ModuleList(
                [LinearRelu(flatten_dim, features_dim) for _ in range(self._n_stacks)]
            )

            # One linear layer for auxiliary observations
            if self._aux_obs_dim != 0:
                self.aux_obs_linear = torch.nn.ModuleList(
                    [
                        LinearRelu(self._aux_obs_dim, aux_obs_features_dim)
                        for _ in range(self._n_stacks)
                    ]
                )

        number_of_learnable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(
            "Initialised OctreeCnnFeaturesExtractor with "
            f"{number_of_learnable_parameters} parameters"
        )
        if verbose:
            print(self)

    def forward(self, obs):
        """
        Note: input octree must be batch of octrees (created with ocnn)
        """

        octree = obs[0]
        aux_obs = obs[1]

        if not self._separate_networks_for_stacks:

            # Extract features from the octree at the finest depth
            data = ocnn.octree_property(octree, "feature", self._depth)

            # Make sure the number of input channels matches the argument passed to constructor
            assert (
                data.size(1) == self._channels_in
            ), f"Input octree has invalid number of channels. Got {data.size(1)}, expected {self._channels_in}"

            # Pass the data through all convolutional and polling layers
            for i in range(self._n_convs):
                data = self.convs[i](data, octree)
                data = self.pools[i](data, octree)

            # Last convolution at full_depth
            if self._full_depth_conv1d:
                # Conv1D
                data = self.full_depth_conv(data)
            else:
                # Conv3D
                data = self.full_depth_conv(data, octree)

            # Convert octree at full_depth into a voxel grid
            data = self.octree2voxel(data)

            # Flatten into a feature vector
            data = self.flatten(data)

            # Feed through the last linear layer
            data = self.linear(data)

            # Get a view that merges stacks into a single feature vector (original batches remain separated)
            data = data.view(-1, self._n_stacks * data.shape[-1])

            if self._aux_obs_dim != 0:
                # Feed the data through linear layer
                aux_data = self.aux_obs_linear(aux_obs.view(-1, self._aux_obs_dim))
                # Get a view that merges aux feature stacks into a single feature vector (original batches remain separated)
                aux_data = aux_data.view(
                    -1, self._n_stacks * self._aux_obs_features_dim
                )
                # Concatenate auxiliary data
                data = torch.cat((data, aux_data), dim=1)

        else:

            # Extract features from the octree at the finest depth
            data = [
                ocnn.octree_property(octree[i], "feature", self._depth)
                for i in range(self._n_stacks)
            ]

            for i in range(self._n_stacks):

                # Pass the data through all convolutional and polling layers
                for j in range(self._n_convs):
                    data[i] = self.convs[i][j](data[i], octree[i])
                    data[i] = self.pools[i][j](data[i], octree[i])

                # Last convolution at full_depth
                if self._full_depth_conv1d:
                    # Conv1D
                    data[i] = self.full_depth_conv[i](data[i])
                else:
                    # Conv3D
                    data[i] = self.full_depth_conv[i](data[i], octree[i])

                # Convert octree at full_depth into a voxel grid
                data[i] = self.octree2voxel[i](data[i])

                # Flatten into a feature vector
                data[i] = self.flatten[i](data[i])

                # Feed through the last linear layer
                data[i] = self.linear[i](data[i])

                if self._aux_obs_dim != 0:
                    # Feed the data through linear layer
                    aux_data = self.aux_obs_linear[i](aux_obs[:, i, :])
                    # Concatenate auxiliary data
                    data[i] = torch.cat((data[i], aux_data), dim=1)

            # Concatenate with other stacks
            data = torch.cat(data, dim=1)

        return data
