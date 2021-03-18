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
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 depth: int = 5,
                 full_depth: int = 2,
                 channels_in: int = 3,
                 channel_multiplier: int = 4,
                 features_dim: int = 512,
                 fast_conv: bool = True,
                 batch_normalization: bool = False):

        # Chain up parent constructor
        super(OctreeCnnFeaturesExtractor, self).__init__(observation_space,
                                                         features_dim)

        self._depth = depth
        self._channels_in = channels_in

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

        # Layer that converts octree at depth=full_depth into a full voxel grid (zero padding) such that it has a fixed size
        self.octree2voxel = ocnn.FullOctree2Voxel(full_depth)
        # Layer that flattens the voxel grid into a feature vector, this is the last layer of feature extractor that should feed into FC layers
        self.flatten = torch.nn.Flatten()

        # The number of flattened features equals the channels at full_depth, times the number of cells in its voxel grid (O=3n)
        flatten_dim = channels[-1]*(2 ** (3 * full_depth))
        # Linear layer of the extractor (and the last layer)
        self.linear = torch.nn.Sequential(torch.nn.Linear(flatten_dim, features_dim),
                                          torch.nn.ReLU())

    def forward(self, octree):
        """
        Note: input octree must be batch of octrees (created with ocnn)
        """

        # Extract features from the octree at the finest depth
        data = ocnn.octree_property(octree, 'feature', self._depth)

        # Make sure the number of input channels matches the argument passed to constructor
        assert data.size(1) == self._channels_in, \
            f"Input octree has invalid number of channels. Got {data.size(2)}, expected {self._channels_in}"

        # Pass the data through all convolutional and polling layers
        for i in range(len(self.convs)):
            data = self.convs[i](data, octree)
            data = self.pools[i](data, octree)

        # Convert octree at full_depth into a voxel grid
        data = self.octree2voxel(data)

        # Flatten into a feature vector
        data = self.flatten(data)

        # Feed through linear layer
        data = self.linear(data)

        return data
