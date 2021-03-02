from typing import Tuple
from scipy.spatial.transform import Rotation
import sensor_msgs
import geometry_msgs
import numpy
import open3d
import struct
import pyoctree

__POINT_FIELD_DTYPES = {}
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.INT8] = ('b', 1)
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.UINT8] = ('B', 1)
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.INT16] = ('h', 2)
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.UINT16] = ('H', 2)
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.INT32] = ('i', 4)
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.UINT32] = ('I', 4)
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.FLOAT32] = ('f', 4)
__POINT_FIELD_DTYPES[sensor_msgs.msg.PointField.FLOAT64] = ('d', 8)


def pointcloud2_to_open3d(ros_point_cloud2: sensor_msgs.msg.PointCloud2,
                          include_rgb: bool = True,
                          fix_uint8_rgb: bool = True) -> open3d.geometry.PointCloud:

    # Assert that all fields are present and ordered as they should be
    assert('x' == ros_point_cloud2.fields[0].name)
    assert('y' == ros_point_cloud2.fields[1].name)
    assert('z' == ros_point_cloud2.fields[2].name)
    if include_rgb:
        assert('rgb' == ros_point_cloud2.fields[3].name)
        if fix_uint8_rgb:
            # TODO: There is an issue somewhere in Ignition Gazebo rgbd_camera or
            # ros_ign conversion to PointCloud2 (incorrect datatype, count and it has extra length)
            ros_point_cloud2.fields[3].datatype = sensor_msgs.msg.PointField.UINT8
            ros_point_cloud2.fields[3].count = 3

    # Create output Open3D PointCloud
    open3d_pc = open3d.geometry.PointCloud()

    # Parse points from PointCloud2 into Open3D PointCloud
    if include_rgb:
        # Note: Color stored as BGR instead of RGB
        for (x, y, z, b, g, r) in _read_points(ros_point_cloud2,
                                               field_names=('x', 'y', 'z',
                                                            'rgb'),
                                               only_finite=True):
            open3d_pc.points.append([x, y, z])
            open3d_pc.colors.append([r / 255, g / 255, b / 255])
    else:
        for xyz in _read_points(ros_point_cloud2,
                                field_names=('x', 'y', 'z'),
                                only_finite=True):
            open3d_pc.points.append(xyz)

    # Return the converted Open3D PointCloud
    return open3d_pc


def _read_points(cloud, field_names=None, only_finite=True):
    # Original source: http://docs.ros.org/en/melodic/api/sensor_msgs/html/point__cloud2_8py_source.html

    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    unpack_from = struct.Struct(fmt).unpack_from

    if only_finite:
        for v in range(cloud.height):
            offset = cloud.row_step * v
            for u in range(cloud.width):
                p = unpack_from(cloud.data, offset)
                has_nan_or_inf = False
                for pv in p:
                    if not numpy.isfinite(pv):
                        has_nan_or_inf = True
                        break
                if not has_nan_or_inf:
                    yield p
                offset += cloud.point_step
    else:
        for v in range(cloud.height):
            offset = cloud.row_step * v
            for u in range(cloud.width):
                yield unpack_from(cloud.data, offset)
                offset += cloud.point_step


def _get_struct_fmt(is_bigendian, fields, field_names=None):

    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in __POINT_FIELD_DTYPES:
            print("Skipping unknown PointField datatype [%d]" % field.datatype)
        else:
            datatype_fmt, datatype_length = __POINT_FIELD_DTYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def transform_to_matrix(transform: geometry_msgs.msg.Transform) -> numpy.ndarray:

    transform_matrix = numpy.zeros((4, 4))
    transform_matrix[3, 3] = 1.0

    transform_matrix[0:3, 0:3] = \
        open3d.geometry.get_rotation_matrix_from_quaternion([transform.rotation.w,
                                                             transform.rotation.x,
                                                             transform.rotation.y,
                                                             transform.rotation.z])
    transform_matrix[0, 3] = transform.translation.x
    transform_matrix[1, 3] = transform.translation.y
    transform_matrix[2, 3] = transform.translation.z

    return transform_matrix


def open3d_point_cloud_to_octree_points(open3d_point_cloud: open3d.geometry.PointCloud,
                                        include_color: bool = False) -> pyoctree.Points:

    octree_points = pyoctree.Points()
    octree_points.set_points(
        # XYZ points
        numpy.reshape(numpy.asarray(open3d_point_cloud.points), -1),
        # Normals
        numpy.reshape(numpy.asarray(open3d_point_cloud.normals), -1),
        # Other features, e.g. color
        (numpy.reshape(numpy.asarray(open3d_point_cloud.colors), - 1)
         if include_color else []),
        # Labels - not used
        []
    )

    return octree_points


def orientation_6d_to_quat(v1: Tuple[float, float, float],
                           v2: Tuple[float, float, float]) -> Tuple[float, float, float, float]:

    # Normalize vectors
    col1 = v1 / numpy.linalg.norm(v1)
    col2 = v2 / numpy.linalg.norm(v2)

    # Find their orthogonal vector via cross product
    col3 = numpy.cross(col1, col2)

    # Stack into rotation matrix as columns, convert to quaternion and return
    quat_xyzw = Rotation.from_matrix(numpy.array([col1,
                                                  col2,
                                                  col3]).T).as_quat()
    return quat_xyzw


def orientation_quat_to_6d(quat_xyzw: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float],
                                                                                  Tuple[float, float, float]]:

    # Convert quaternion into rotation matrix
    rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()

    # Return first two columns (already normalised)
    return (tuple(rot_mat[:, 0]), tuple(rot_mat[:, 1]))
