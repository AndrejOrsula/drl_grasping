#!/usr/bin/env python3

from drl_grasping.perception import OctreeCreator
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import ocnn
import rclpy
import torch

# Note: Environment must already be running in background before starting this test. It can easily be combined with ./test_env.py (with env that has octree observations)


class OctreeCreatorTest(Node):
    def __init__(self):

        # Initialise node
        try:
            rclpy.init()
        except:
            if not rclpy.ok():
                import sys
                sys.exit("ROS 2 could not be initialised")
        Node.__init__(self, "octree_creator_test")

        self.__point_cloud_sub = self.create_subscription(PointCloud2,
                                                          "rgbd_camera/points",
                                                          self.point_cloud_callback, 1)

        self.octree_creator = OctreeCreator()

    def point_cloud_callback(self, msg):
        """
        Callback for getting point cloud.
        """

        octree = self.octree_creator(msg)

        self.test_simple(octree=octree)
        self.test_padded(octree=octree)

    def test_simple(self, octree: torch.Tensor):

        # Configuration
        depth = self.octree_creator._depth
        channels = 6 if self.octree_creator._include_color else 3
        num_outputs = 5

        # Create batch from the octree and move it to VRAM (it has to be in VRAM for the next step)
        octree_batch = ocnn.octree_batch([octree]).cuda()

        # Extract features from the octree
        data = ocnn.octree_property(octree_batch, 'feature', depth).cuda()
        assert data.size(1) == channels

        # Test simple convolution
        conv1 = ocnn.OctreeConv(depth, channels, num_outputs)
        conv1.cuda()
        out1 = conv1(data, octree_batch)

        # Test fast convolution
        conv2 = ocnn.OctreeConvFast(depth, channels, num_outputs)
        conv2.cuda()
        out2 = conv2(data, octree_batch)

    def test_padded(self, octree: torch.Tensor):

        # Configuration
        depth = self.octree_creator._depth
        channels = 6 if self.octree_creator._include_color else 3
        num_outputs = 5

        # Pad octree to a specific, fixed length
        print(f'Original size: {octree.shape}')
        padded_size = 1000000
        octree_padded = torch.nn.ConstantPad1d(
            (0, padded_size - octree.shape[0]), 0)(octree)

        # Create batch from the octree and move it to VRAM (it has to be in VRAM for the next step)
        octree_batch = ocnn.octree_batch([octree_padded]).cuda()

        # Extract features from the octree
        data = ocnn.octree_property(octree_batch, 'feature', depth).cuda()
        assert data.size(1) == channels

        # Test simple convolution
        conv1 = ocnn.OctreeConv(depth, channels, num_outputs)
        conv1.cuda()
        out1 = conv1(data, octree_batch)

        # Test fast convolution
        conv2 = ocnn.OctreeConvFast(depth, channels, num_outputs)
        conv2.cuda()
        out2 = conv2(data, octree_batch)


def main(args=None):
    rclpy.init(args=args)

    rclpy.spin(OctreeCreatorTest())

    rclpy.shutdown()


if __name__ == "__main__":
    main()
