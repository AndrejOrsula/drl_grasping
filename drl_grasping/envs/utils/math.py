from typing import Tuple


def quat_mul(quat_0: Tuple[float, float, float, float],
             quat_1: Tuple[float, float, float, float],
             xyzw: bool = True) -> Tuple[float, float, float, float]:
    """
    Multiply two quaternions
    """
    if xyzw:
        x0, y0, z0, w0 = quat_0
        x1, y1, z1, w1 = quat_1
        return (x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)
    else:
        w0, x0, y0, z0 = quat_0
        w1, x1, y1, z1 = quat_1
        return (-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)
