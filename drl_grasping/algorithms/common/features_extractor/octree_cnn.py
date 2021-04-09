from drl_grasping.algorithms.common.features_extractor.modules import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import ocnn
import torch


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

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 depth: int = 5,
                 full_depth: int = 2,
                 channels_in: int = 4,
                 channel_multiplier: int = 16,
                 full_depth_conv1d: bool = False,
                 full_depth_channels: int = 8,
                 features_dim: int = 128,
                 aux_obs_dim: int = 0,
                 fast_conv: bool = True,
                 batch_normalization: bool = True,
                 verbose: bool = False):

        self._depth = depth
        self._channels_in = channels_in
        self._aux_obs_dim = aux_obs_dim
        self._verbose = verbose
        self.n_stacks = observation_space.shape[0]

        # Chain up parent constructor
        super(OctreeCnnFeaturesExtractor, self).__init__(observation_space,
                                                         self.n_stacks*(features_dim+aux_obs_dim))

        # Channels ordered as [channels_in, depth, depth-1, ..., full_depth]
        # I.e [channels_in, channel_multiplier*1, channel_multiplier*2, channel_multiplier*4, channel_multiplier*8,...]
        channels = [channel_multiplier*(2**i) for i in range(depth-full_depth)]
        channels.insert(0, channels_in)

        # Create all Octree convolution and pooling layers in depth-descending order [depth, depth-1, ..., full_depth]
        # Input to the first conv layer is the input Octree at depth=depth
        # Output from the last pool layer is feature map at depth=full_depth
        if fast_conv:
            if batch_normalization:
                OctreeConv = OctreeConvFastBnRelu
            else:
                OctreeConv = OctreeConvFastRelu
        else:
            if batch_normalization:
                OctreeConv = OctreeConvBnRelu
            else:
                OctreeConv = OctreeConvRelu
        OctreePool = ocnn.OctreeMaxPool
        self.convs = torch.nn.ModuleList([OctreeConv(depth-i, channels[i], channels[i+1])
                                          for i in range(depth-full_depth)])
        self.pools = torch.nn.ModuleList([OctreePool(depth-i)
                                          for i in range(depth-full_depth)])

        # Last convolution at depth=full_depth, which is not follewed by pooling
        # This layer is used to significantly reduce the channels, decresing number of parameters required in the next linear layer(s)
        self._full_depth_conv1d = full_depth_conv1d
        if self._full_depth_conv1d:
            # Use 1D convolution (Conv1D instead of linear is used here to preserve spatial information)
            if batch_normalization:
                OctreeConv1D = OctreeConv1x1BnRelu
            else:
                OctreeConv1D = OctreeConv1x1Relu
            self.full_depth_conv = OctreeConv1D(channels[-1],
                                                full_depth_channels)
        else:
            # Use 3D convolution (same as previous modules)
            self.full_depth_conv = OctreeConv(full_depth,
                                              channels[-1],
                                              full_depth_channels)

        # Layer that converts octree at depth=full_depth into a full voxel grid (zero padding) such that it has a fixed size
        self.octree2voxel = ocnn.FullOctree2Voxel(full_depth)
        full_depth_voxel_count = 2**(3*full_depth)

        # Layer that flattens the voxel grid into a feature vector
        self.flatten = torch.nn.Flatten()
        flatten_dim = full_depth_channels*full_depth_voxel_count

        # Last linear layer of the extractor, applied to all (flattened) voxels at full depth
        if batch_normalization:
            LineadModule = LinearBnRelu
        else:
            LineadModule = LinearRelu
        self.linear = LineadModule(flatten_dim, features_dim)

        # One linear layer for auxiliary observations
        if self._aux_obs_dim != 0:
            self.aux_obs_linear = LineadModule(
                self._aux_obs_dim, self._aux_obs_dim)

        number_of_learnable_parameters = sum(p.numel() for p in self.parameters()
                                             if p.requires_grad)
        print("Initialised OctreeCnnFeaturesExtractor with "
              f"{number_of_learnable_parameters} parameters")
        if verbose:
            print(self)

    def forward(self, octree):
        """
        Note: input octree must be batch of octrees (created with ocnn)
        """

        aux_obs = octree['aux_obs']
        octree = octree['octree']

        # Extract features from the octree at the finest depth
        data = ocnn.octree_property(octree, 'feature', self._depth)

        # Make sure the number of input channels matches the argument passed to constructor
        assert data.size(1) == self._channels_in, \
            f"Input octree has invalid number of channels. Got {data.size(1)}, expected {self._channels_in}"

        # Pass the data through all convolutional and polling layers
        for i in range(len(self.convs)):
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
        data = data.view(-1, self.n_stacks*data.shape[-1])

        if self._aux_obs_dim != 0:
            # Get a view that merges aux feature stacks into a single feature vector (original batches remain separated)
            aux_obs = aux_obs.view(-1, self.n_stacks*self._aux_obs_dim)
            # Feed the data through linear layer
            aux_data = self.linear(aux_obs)
            # Concatenate auxiliary data
            data = torch.cat((data, aux_data), dim=1)

        return data
