import numpy as np
import tensorflow as tf
from collections import namedtuple


INVALID_GROUP = 0


Outlines = namedtuple('Outlines', ['points', 'groups', 'all_groups'])


def tf_rotation_matrix(angles):
    assert angles.shape is not None
    with tf.name_scope('rotation_matrix'):
        sin = tf.sin(angles)
        cos = tf.cos(angles)
        return tf.stack([
            tf.stack([cos, -sin], axis=-1),
            tf.stack([sin, cos], axis=-1)
        ], axis=-2)


def tf_scale_matrix(scale_x, scale_y):
    assert scale_x.shape is not None
    with tf.name_scope('scale_matrix'):
        zeros = tf.zeros_like(scale_x)
        return tf.stack([
            tf.stack([scale_x, zeros], axis=-1),
            tf.stack([zeros, scale_y], axis=-1)
        ], axis=-2)


def tf_transform_points(batch_points, transform):
    with tf.name_scope('transform_points'):
        # [batch, n_pts, n_feat]
        assert len(batch_points.shape) == 3
        n_pts = tf.shape(batch_points)[1]

        # [batch, 2, 2]
        assert len(transform.shape) == 3
        assert transform.shape[1] == transform.shape[2] == 2
        # Dimensions with None do not pass comparison directly
        assert transform.shape[0].value == batch_points.shape[0].value

        # [batch, 1, 2, 2]
        transform = tf.expand_dims(transform, axis=1)

        # [batch, n_pts, 2, 2]
        transform = tf.tile(transform, (1, n_pts, 1, 1))

        return tf.squeeze(
            tf.matmul(transform, tf.expand_dims(batch_points, axis=-1)),
            axis=-1
        )


def tf_group_hot(values, groups, all_groups):
    with tf.name_scope('group_hot'):
        while len(groups.shape) < len(values.shape):
            groups = tf.expand_dims(groups, axis=-1, name='group_expand')
        return tf.concat([
            values * tf.cast(tf.equal(groups, g), values.dtype)
            for g in all_groups
        ], axis=-1)


def points_4d(points):
    assert len(points.shape) in (3, 4)
    if len(points.shape) == 3:
        with tf.name_scope('points_4d'):
            points = tf.expand_dims(points, axis=-2)
    assert len(points.shape) == 4
    return points


def groups_4d(groups):
    assert len(groups.shape) in (2, 4)
    if len(groups.shape) == 2:
        with tf.name_scope('groups_4d'):
            groups = tf.expand_dims(groups, axis=-1)
            groups = tf.expand_dims(groups, axis=-1)
    assert len(groups.shape) == 4
    return groups


def get_pad(n_neighbors):
    assert n_neighbors % 2 == 1
    return (n_neighbors - 1) // 2


# @TESTED: 2018-11-
def group_conv(groups, all_groups, n_neighbors, strides_neighbors=1,
               enable_group_crossing=True, name='group_conv'):
    with tf.name_scope(name):
        # Optimize the common case
        if n_neighbors == 1:
            return groups[:, ::strides_neighbors]

        groups_kernel = tf.ones([n_neighbors, 1, 1, 1], dtype=tf.float32)
        groups_strides = [strides_neighbors, 1, 1, 1]

        src_groups = groups
        neighbor_pad = get_pad(n_neighbors)
        if neighbor_pad > 0:
            with tf.name_scope('padding'):
                batch_size = tf.shape(groups)[0]
                group_pad = INVALID_GROUP * tf.ones(
                    shape=[batch_size, neighbor_pad, 1, 1],
                    dtype=groups.dtype
                )
                groups = tf.concat([group_pad, groups, group_pad], axis=1)

        new_group_mask = INVALID_GROUP * tf.ones_like(src_groups)

        if n_neighbors > 1 and not enable_group_crossing:
            for i, g in enumerate(all_groups):
                group_validity = tf.equal(
                    float(n_neighbors),
                    tf.nn.conv2d(
                        tf.cast(tf.equal(groups, g), dtype=tf.float32),
                        filter=groups_kernel, strides=groups_strides,
                        padding='VALID', name='group_%d_conv' % i
                    )
                )
                new_group_mask = tf.where(group_validity, src_groups, new_group_mask)
        else:
            ok = tf.equal(
                float(n_neighbors),
                tf.nn.conv2d(
                    tf.cast(tf.not_equal(groups, INVALID_GROUP), dtype=tf.float32),
                    filter=groups_kernel, strides=groups_strides,
                    padding='VALID', name='group_conv'
                )
            )
            new_group_mask = tf.where(ok, src_groups, new_group_mask)

        return new_group_mask


def test_group_conv():
    print('Testing group_conv')

    if not tf.executing_eagerly():
        tf.enable_eager_execution()

    groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0])

    def run_group_conv(groups, all_groups, n_neighbors, enable_group_crossing):
        groups = groups_4d(np.array([groups]))

        groups2 = group_conv(
            groups, all_groups=all_groups, n_neighbors=n_neighbors,
            enable_group_crossing=enable_group_crossing
        )
        groups2 = tf.squeeze(tf.squeeze(groups2, axis=-1), axis=-1)
        groups2 = groups2.numpy()

        return groups2

    assert np.array_equiv(
        run_group_conv(groups, all_groups=[1, 2],
                       n_neighbors=3, enable_group_crossing=True),
        [0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
    )

    assert np.array_equiv(
        run_group_conv(groups, all_groups=[1, 2],
                       n_neighbors=3, enable_group_crossing=True),
        [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0]
    )

    assert np.array_equiv(
        run_group_conv(groups, all_groups=[1, 2],
                       n_neighbors=1, enable_group_crossing=True),
        groups
    )

    assert np.array_equiv(
        run_group_conv(groups, all_groups=[1, 2],
                       n_neighbors=1, enable_group_crossing=True),
        groups
    )


def point_conv(points, n_neighbors, n_out, strides_neighbors=1,
               activation=tf.nn.relu, name='point_conv', weights=None,
               regularizer=None):
    with tf.variable_scope(name):
        # batch X n_points (height) X 1 (width) X features
        n_in = points.shape[-1].value
        assert n_in is not None

        neighbor_pad = get_pad(n_neighbors)

        if neighbor_pad > 0:
            with tf.name_scope('padding'):
                batch_size = tf.shape(points)[0]
                pt_pad = tf.zeros(
                    shape=[batch_size, neighbor_pad, 1, n_in],
                    dtype=points.dtype
                )
                points = tf.concat([pt_pad, points, pt_pad], axis=1)

        result = tf.layers.conv2d(points, n_out, kernel_size=[n_neighbors, 1],
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  strides=[strides_neighbors, 1],
                                  padding='VALID')

        if activation is not None:
            result = activation(result)

        return result


def outline_conv(outlines: Outlines, n_neighbors, n_out, strides_neighbors=1,
                 enable_group_crossing=True, activation=tf.nn.relu,
                 name='outline_conv', weights=None, regularizer=None):
    with tf.variable_scope(name):
        points = point_conv(
            points=outlines.points, n_out=n_out,
            n_neighbors=n_neighbors, strides_neighbors=strides_neighbors,
            activation=activation, regularizer=regularizer
        )

        groups = group_conv(
            groups=outlines.groups, all_groups=outlines.all_groups,
            n_neighbors=n_neighbors, strides_neighbors=strides_neighbors,
            enable_group_crossing=enable_group_crossing
        )

        return outlines._replace(points=points, groups=groups)


def outline_normalize(outlines: Outlines):
    def single_normalize(args):
        pts, grp = args
        # Squeeze the group back from it's image form
        valid = tf.not_equal(grp[:, 0, 0], INVALID_GROUP)
        center = tf.reduce_mean(
            tf.boolean_mask(
                pts, valid, axis=0
            ),
            axis=0, keepdims=True
        )
        return tf.cond(
            tf.reduce_any(valid),
            lambda: pts - center,
            lambda: pts
        )

    with tf.name_scope('outline_normalize'):
        return outlines._replace(
            points=tf.map_fn(
                single_normalize, (outlines.points, outlines.groups),
                dtype=outlines.points.dtype
            )
        )


def test_outline_normalize():
    print('Testing outline_normalize')

    if not tf.executing_eagerly():
        tf.enable_eager_execution()

    groups = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0]
    pts =    [8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 9, 9]

    def run_outline_normalize(points, groups, all_groups):
        groups = groups_4d(np.array([groups]))
        points = points_4d(np.reshape([points], (1, -1, 1)))

        o = Outlines(points, groups, all_groups)
        o = outline_normalize(o)

        points = tf.reshape(o.points, [-1])
        groups = tf.reshape(o.groups, [-1])

        return points.numpy(), groups.numpy()

    pts2, groups2 = run_outline_normalize(pts, groups, [1, 2])

    assert np.array_equiv(pts2 + 7, pts)
    assert np.array_equiv(groups2, groups)


def outline_max(outlines: Outlines):
    def single_max(args):
        pts, grp = args
        # Squeeze the group back from it's image form
        valid = tf.not_equal(grp[:, 0, 0], INVALID_GROUP)

        return tf.reduce_max(
            tf.cond(tf.reduce_any(valid),
                    lambda: tf.boolean_mask(pts, valid, axis=0),
                    lambda: tf.zeros_like(pts)),
            axis=0
        )

    with tf.name_scope('outline_max'):
        return tf.map_fn(
            single_max, (outlines.points, outlines.groups),
            dtype=outlines.points.dtype
        )


def tf_angle_between(vec1s, vec2s):
    import tensorflow as tf
    x1, y1 = vec1s[..., 0], vec1s[..., 1]
    x2, y2 = vec2s[..., 0], vec2s[..., 1]
    # See https://stackoverflow.com/a/16544330
    dot = x1 * x2 + y1 * y2  # dot product between [x1, y1] and [x2, y2]
    det = x1 * y2 - y1 * x2  # determinant
    return tf.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)


def group_jitter(groups, all_groups, radius):
    with tf.name_scope('group_jitter'):
        inner_group_vals = tf.not_equal(
            group_conv(
                groups=groups,
                all_groups=all_groups,
                n_neighbors=2 * radius + 1,
                enable_group_crossing=True
            ),
            INVALID_GROUP
        )

        jitter_offset = tf.random_uniform(
            shape=(tf.shape(groups)[0],),
            minval=-radius, maxval=+radius,
            dtype=tf.int32
        )

        def jitter_single(args):
            grp, inner_vals, offset = args
            return tf.where(
                inner_vals,
                tf.manip.roll(grp, shift=offset, axis=0),
                grp
            )

        return tf.map_fn(
            jitter_single, (groups, inner_group_vals, jitter_offset),
            dtype=groups.dtype
        )


# @ TESTED, compatiable with tf.manip.roll
def point_shift(batch_points, batch_counts, shift):
    shape = tf.shape(batch_points)
    batch_size = shape[0]
    n_pts = shape[1]

    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=-1),
        (1, n_pts)
    )
    point_indices = tf.tile(
        tf.expand_dims(tf.range(n_pts), axis=0),
        (batch_size, 1)
    )
    assert len(batch_counts.shape) == 1
    batch_counts = tf.expand_dims(batch_counts, axis=-1)
    valid_mask = point_indices < batch_counts
    shifted = (point_indices + shift) % batch_counts
    valid_shifted = tf.where(
        valid_mask, shifted, point_indices
    )
    return tf.gather_nd(
        batch_points, tf.stack((batch_indices, valid_shifted), axis=-1)
    )


def points_angle(points, counts, radius, avg_point_distance, is_training=None, keep_prob=0.3):
    with tf.name_scope('points_angle'):
        shape = tf.shape(points)
        batch_size, n_pts = shape[0], shape[1]

        prevs = point_shift(points, counts, shift=-radius)
        to_prevs = prevs - points
        to_prevs_n = tf.linalg.norm(to_prevs, axis=-1)

        nexts = point_shift(points, counts, shift=radius)
        to_nexts = nexts - points
        to_nexts_n = tf.linalg.norm(to_nexts, axis=-1)

        angles = tf_angle_between(vec1s=to_prevs, vec2s=to_nexts)
        shape = tf.shape(angles)
        if avg_point_distance is not None:
            valid_dist = avg_point_distance * (tf.cast(radius, dtype=tf.float32) + 0.5) * 1.01
            bad_mask1 = tf.logical_or(
                to_prevs_n > valid_dist,
                to_nexts_n > valid_dist,
            )
        else:
            bad_mask1 = tf.fill(dims=tf.shape(to_prevs_n), value=False)
        bad_mask2 = tf.expand_dims(tf.range(n_pts), axis=0) >= tf.expand_dims(counts, -1)
        bad_mask = tf.logical_or(bad_mask1, bad_mask2)

        filler = tf.fill(shape, tf.constant(np.pi, dtype=angles.dtype))

        result = angles
        result = tf.where(
            bad_mask, filler, result
        )
        result = tf.where(
            tf.is_nan(result), filler, result
        )
        if is_training is not None and keep_prob < 1:
            before_drop = result
            drop_mask = tf.random_uniform(shape=shape) > keep_prob
            result = tf.cond(is_training,
                             lambda: before_drop,
                             lambda: tf.where(drop_mask, filler, result))
        return result


def outline_angle(outlines: Outlines, radius, enable_group_crossing=True):
    with tf.name_scope('outline_angle'):
        pts = outlines.points
        angles = tf_angle_between(
            vec1s=(tf.manip.roll(pts, shift=+radius, axis=1) - pts),
            vec2s=(tf.manip.roll(pts, shift=-radius, axis=1) - pts)
        )
        angles = tf.expand_dims(angles, axis=-2)
        assert len(angles.shape) == 4
        new_groups = group_conv(
            groups=outlines.groups,
            all_groups=outlines.all_groups,
            n_neighbors=radius * 2 + 1,
            enable_group_crossing=enable_group_crossing
        )
        return Outlines(
            points=angles,
            groups=new_groups,
            all_groups=outlines.all_groups
        )


def dropout(val, is_training, name='drop', keep_prob=0.7):
    return tf.cond(is_training,
                   lambda: tf.nn.dropout(val, keep_prob=keep_prob),
                   lambda: val,
                   name=name)


def outline_dropout(o, is_training, name='drop', keep_prob=0.7):
    return o._replace(points=dropout(o.points, is_training, name, keep_prob))


def outline_net(points, groups, angle_rep, all_groups, is_training, n_classes, reg_scale=0):
    from c3d.shape2.gen_pcl_dataset import ANGLE_SAMPLE_DIST, TRAIN_SAMPLE_POINTS_BY_DIST
    points = points_4d(points)
    groups = groups_4d(groups)
    debug = {}

    outlines = Outlines(
        points=points,
        groups=groups,
        all_groups=all_groups
    )

    angles = outline_angle(
        outlines, radius=int(np.ceil(ANGLE_SAMPLE_DIST / TRAIN_SAMPLE_POINTS_BY_DIST)),
        enable_group_crossing=True
    )
    angles = angles._replace(points=angle_rep(angles.points))

    regularizer = None if reg_scale <= 0 else tf.contrib.layers.l2_regularizer(reg_scale)

    def make_features(outlines, name):
        with tf.variable_scope(name):
            net = outlines
            net = net._replace(
                points=tf_group_hot(values=outlines.points, groups=outlines.groups, all_groups=outlines.all_groups)
            )
            debug[name + '_l_in'] = net
            print(net.points)

            net = outline_normalize(net)
            debug[name + '_l_in_norm'] = net
            print(net.points)

            net = outline_conv(net, n_neighbors=1, n_out=64,
                               enable_group_crossing=True,
                               regularizer=regularizer,
                               activation=tf.nn.relu, name='conv1')
            #net = outline_dropout(net, is_training, 'conv1_drop')
            debug[name + '_l_1'] = net
            print(net.points)

            net = outline_conv(net, n_neighbors=5, n_out=64,
                               enable_group_crossing=True,
                               regularizer=regularizer,
                               activation=tf.nn.relu, name='conv2')
            #net = outline_dropout(net, is_training, 'conv2_drop')
            debug[name + '_l_2'] = net
            print(net.points)

            net = outline_conv(net, n_neighbors=5, n_out=64,
                               enable_group_crossing=True,
                               regularizer=regularizer,
                               activation=tf.nn.relu, name='conv3')
            #net = outline_dropout(net, is_training, 'conv3_drop')
            debug[name + '_l_3'] = net
            print(net.points)

            net = outline_conv(net, n_neighbors=5, n_out=128,
                               enable_group_crossing=True,
                               regularizer=regularizer,
                               activation=tf.nn.relu, name='conv4')
            #net = outline_dropout(net, is_training, 'conv4_drop')
            debug[name + '_l_4'] = net
            print(net.points)
            return net

    pt_results = make_features(outlines, 'point_outlines')
    angle_results = make_features(angles, 'angle_outlines')

    net = angle_results._replace(
        points=tf.concat((pt_results.points, angle_results.points), axis=-1)
    )
    debug['combine_l_4'] = net

    net = outline_conv(net, n_neighbors=1, n_out=1024,
                       enable_group_crossing=True,
                       regularizer=regularizer,
                       activation=tf.nn.relu, name='conv_combine1')
    #net = outline_dropout(net, is_training, 'conv4_drop')
    print(net)
    debug['combine_l_5'] = net

    net = outline_max(net)
    debug['combine_l_6'] = net
    print(net)

    net = tf.squeeze(net, axis=1)
    features = net

    net = tf.layers.dense(net, 512, activation=tf.nn.relu,
                          name='fc1',
                          bias_regularizer=regularizer,
                          kernel_regularizer=regularizer)
    net = dropout(net, is_training, 'fc1_drop')
    debug['combine_l_7'] = net
    print(net)

    net = tf.layers.dense(net, 256, activation=tf.nn.relu,
                          name='fc2',
                          bias_regularizer=regularizer,
                          kernel_regularizer=regularizer)
    net = dropout(net, is_training, 'fc2_drop')
    debug['combine_l_8'] = net
    print(net)

    net = tf.layers.dense(net, n_classes, activation=None,
                          name='fc3')
    debug['combine_l_9'] = net
    print(net)

    return features, net, debug


def test_suite():
    print('Running tests')
    tf.enable_eager_execution()

    test_group_conv()
    test_outline_normalize()
    print('All tests passed successfully')


if __name__ == '__main__':
    test_suite()
