from collections import namedtuple
import numpy as np
import random


Plane3D = namedtuple('Plane3D', 'A B C D')
Line2D = namedtuple('Line2D', 'A B C')


def z_rot_matrix(angle):
    theta = np.deg2rad(angle)
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([
        [+cos, -sin, 0],
        [+sin, +cos, 0],
        [0, 0,       1],
    ])


def x_rot_matrix(angle):
    theta = np.deg2rad(angle)
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([
        [1,    0,    0],
        [0, +cos, -sin],
        [0, +sin, +cos],
    ])


def rot(matrix, vec):
    return np.matmul(matrix, vec.reshape((3, 1))).reshape(vec.shape)


def generate_fracture_plane(max_radius):
    z_rot = np.random.uniform(0, 90)
    x_rot = np.random.uniform(-20, 20)
    plane_normal = np.array([1, 0, 0])
    plane_normal = rot(z_rot_matrix(z_rot), plane_normal)
    plane_normal = rot(x_rot_matrix(x_rot), plane_normal)
    return Plane3D(A=plane_normal[0], B=plane_normal[1], C=plane_normal[2],
                   D=np.random.uniform(0, max_radius))


def gen_line(slope_angle, height, mean_x=0):
    slope = np.tan(np.deg2rad(slope_angle))
    real_height = height - slope * mean_x
    return Line2D(A=slope, B=-1, C=real_height)


def random_on_interval(start, end):
    length = abs(end - start)
    loc = abs(
        np.random.normal(loc=0, scale=length/4)
    )
    loc = min(length, loc)
    if start < end:
        return start + loc
    else:
        return start - loc


def make_y_range_lines(bottom, top, min_size=10, mean_x=0):
    slope_angles = np.random.uniform(-45, +45, size=(2,))

    chance = random.random()
    if chance < 0.8:  # 80% chance to include the top
        line_bot = gen_line(slope_angle=slope_angles[0],
                            height=random_on_interval(top - min_size, bottom),
                            mean_x=mean_x)
        line_top = gen_line(slope_angle=0, height=top + 1, mean_x=mean_x)
    elif chance < 0.1:  # 10% chance to include the bottom
        line_bot = gen_line(slope_angle=0, height=bottom - 1, mean_x=mean_x)
        line_top = gen_line(slope_angle=slope_angles[0],
                            height=random_on_interval(bottom + min_size, top),
                            mean_x=mean_x)
    elif chance < 0.95:  # 5% chance to have both cut
        size = random_on_interval(min_size, (top - bottom))
        top_height = np.random.uniform(bottom + size, top)
        line_top = gen_line(slope_angle=slope_angles[0],
                            height=top_height,
                            mean_x=mean_x)
        line_bot = gen_line(slope_angle=slope_angles[1],
                            height=top - size,
                            mean_x=mean_x)
    else:  # 5% change to have all
        line_top = gen_line(slope_angle=0, height=top + 1, mean_x=mean_x)
        line_bot = gen_line(slope_angle=0, height=bottom - 1, mean_x=mean_x)

    return line_bot, line_top
