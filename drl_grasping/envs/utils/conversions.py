from typing import Tuple, Union
from scipy.spatial.transform import Rotation
import sensor_msgs
import geometry_msgs
import numpy
import open3d
import pyoctree


def pointcloud2_to_open3d(ros_point_cloud2: sensor_msgs.msg.PointCloud2,
                          include_rgb: bool = True) -> open3d.geometry.PointCloud:

    # Create output Open3D PointCloud
    open3d_pc = open3d.geometry.PointCloud()

    size = ros_point_cloud2.width*ros_point_cloud2.height
    xyz_dtype = '>f4' if ros_point_cloud2.is_bigendian else '<f4'
    xyz = numpy.ndarray(shape=(size, 3),
                        dtype=xyz_dtype,
                        buffer=ros_point_cloud2.data,
                        offset=0,
                        strides=(ros_point_cloud2.point_step, 4))

    valid_points = numpy.isfinite(xyz).any(axis=1)
    open3d_pc.points = open3d.utility.Vector3dVector(
        xyz[valid_points].astype(numpy.float64))

    if include_rgb:
        bgr = numpy.ndarray(shape=(size, 3),
                            dtype=numpy.uint8,
                            buffer=ros_point_cloud2.data,
                            offset=ros_point_cloud2.fields[3].offset,
                            strides=(ros_point_cloud2.point_step, 1))
        open3d_pc.colors = open3d.utility.Vector3dVector(
            (bgr[valid_points][:, [2, 1, 0]] / 255).astype(numpy.float64))

    # Return the converted Open3D PointCloud
    return open3d_pc


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


def quat_to_wxyz(xyzw: Union[numpy.ndarray, Tuple[float, float, float, float]]) -> numpy.ndarray:

    if isinstance(xyzw, tuple):
        return xyzw[3], xyzw[0], xyzw[1], xyzw[2]

    if xyzw.shape != (4,):
        raise ValueError(xyzw)

    return xyzw[[3, 0, 1, 2]]


def quat_to_xyzw(wxyz: Union[numpy.ndarray, Tuple[float, float, float, float]]) -> numpy.ndarray:

    if isinstance(wxyz, tuple):
        return wxyz[1], wxyz[2], wxyz[3], wxyz[0]

    if wxyz.shape != (4,):
        raise ValueError(wxyz)

    return wxyz[[1, 2, 3, 0]]
