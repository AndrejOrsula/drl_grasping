import ocnn
import torch

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


class OctreeConvBnRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1, bn_eps=0.001, bn_momentum=0.01):
        super(OctreeConvBnRelu, self).__init__()
        self.conv = ocnn.OctreeConv(depth,
                                    channel_in,
                                    channel_out,
                                    kernel_size,
                                    stride)
        self.bn = torch.nn.BatchNorm2d(channel_out,
                                       bn_eps,
                                       bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.bn(out)
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


class OctreeConvFastBnRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1, bn_eps=0.001, bn_momentum=0.01):
        super(OctreeConvFastBnRelu, self).__init__()
        self.conv = ocnn.OctreeConvFast(depth,
                                        channel_in,
                                        channel_out,
                                        kernel_size,
                                        stride)
        self.bn = torch.nn.BatchNorm2d(channel_out,
                                       bn_eps,
                                       bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.bn(out)
        out = self.relu(out)
        return out
