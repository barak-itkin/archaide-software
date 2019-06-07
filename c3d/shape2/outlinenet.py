import json
import numpy as np
import tensorflow as tf

import c3d.smartloss
from c3d.datamodel import Outline2
from c3d.shape2.tf_simple_conv import (
    tf_group_hot, points_angle, tf_rotation_matrix, tf_transform_points
)
from pointnet.utils import tf_util
import pointnet.train


class NestedDict:
    def to_json(self):
        return {
            k: (v if not isinstance(v, NestedDict) else v.to_json())
            for k, v in vars(self).items()
        }

    def from_json(self, vals):
        for k, v in vals.items():
            assert hasattr(self, k)
            if isinstance(getattr(self, k), NestedDict):
                v = getattr(self, k).from_json(v)
            setattr(self, k, v)
        return self

    def set_item(self, key, value):
        parts = key.split('.')
        root = self
        for part in parts[:-1]:
            root = getattr(root, part)
        setattr(root, parts[-1], value)

    def set_via_string(self, key, value):
        existing = self.get_item(key)
        if not isinstance(existing, str):
            value = eval(value)
        self.set_item(key, value)

    def get_item(self, key):
        parts = key.split('.')
        root = self
        for part in parts[:-1]:
            root = getattr(root, part)
        return getattr(root, parts[-1])

    def __str__(self):
        return json.dumps(self.to_json(), indent=2, sort_keys=True)


class PointCNNSpec(NestedDict):
    def __init__(self, num_class=None):
        self.num_class = num_class
        self.batch_size = 200

        self.learning_rate_base = 0.01
        self.decay_steps = 8000
        self.decay_rate = 0.5
        self.learning_rate_min = 1e-6

        self.weight_decay = 1e-5

        self.in_features = 2

        self.xconv_param_name = ('K', 'D', 'P', 'C', 'links')
        self.xconv_param_skeleton = [
            (8, 1, -1, 16, []),  # C *= self.in_features
            (12, 2, 256, 32, []),  # C *= self.in_features
            (16, 2, 128, 64, []),  # C *= self.in_features
            (16, 3, 128, 2, [])  # C *= self.num_class
        ]

        self.with_global = True

        self.fc_param_name = ('C', 'dropout_rate')
        self.fc_params_skeleton = [
            (2, 0.0),
            (2, 0.5)
        ]

        self.sampling = 'random'
        self.adam_epsilon = 1e-2

        self.use_extra_features = True
        self.with_X_transformation = True
        self.sorting_method = None

    @property
    def xconv_params(self):
        result = [dict(zip(self.xconv_param_name, xconv_param))
                  for xconv_param in self.xconv_param_skeleton]
        for layer in result[:-1]:
            layer['C'] *= self.in_features
        result[-1]['C'] *= self.num_class
        return result

    @property
    def fc_params(self):
        result = [dict(zip(self.fc_param_name, fc_param))
                  for fc_param in self.fc_params_skeleton]
        for layer in result:
            layer['C'] *= self.num_class
        return result

    def make_lr(self, batch_num):
        lr_exp_op = tf.train.exponential_decay(
            self.learning_rate_base, batch_num, self.decay_steps,
            self.decay_rate, staircase=True
        )
        return tf.maximum(lr_exp_op, self.learning_rate_min)


class LearningSpec(NestedDict):
    def __init__(self):
        self.lr_init = 1e-6
        self.lr_decay_step = 20000
        self.lr_decay_rate = 0.7
        self.batch_size = 100

        self.use_smartloss = True
        self.smartloss_input_alpha = 6
        self.smartloss_output_alpha = 5
        self.smartloss_out_always = False
        self.smartloss_fp = True
        self.smartloss_hit_k = 1
        self.smartloss_input_k = 1
        self.smartloss_output_k = 1
        self.smartloss_use_in = True
        self.smartloss_use_out = True
        self.smartloss_out_shift = 1
        self.smartloss_normalize = 1
        self.smartloss_gamma = 0.8
        self.smartloss_accum_batches = 50

        self.focal_loss = False
        self.focal_loss_gamma = 2

    def make_lr(self, batch_num):
        return pointnet.train.get_learning_rate(
            batch_num, base_rate=self.lr_init,
            batch_size=self.batch_size,
            decay_step=self.lr_decay_step,
            decay_rate=self.lr_decay_rate,
        )


class DataSpec(NestedDict):
    def __init__(self):
        # Scale all points from mm to 0.1 meters
        self.global_scale = 0.01
        self.augment_train = True
        # Additional scale to apply to SVG inputs
        self.svg_scale = 1

        # Maximal noise range (2 * radius) for each point,
        # in the original input scale
        self.max_noise_range = 1

        # Add a random scale augmentation
        self.scale_center = 1.3
        self.scale_std = 0.5
        self.scale_min = 0.7

        # Add a random horizontal scale augmentation, simulating fractures
        # that are not exactly parallel to the image plane.
        self.hscale_center = 1
        self.hscale_std = 0.2
        self.hscale_min = 0.7

        # Add a random rotation, as fractures aren't always exactly
        # straightened up.
        self.rot_center = 0
        self.rot_std = np.deg2rad(10)

        # Use random attractors for distorting outlines further.
        self.attractor_distort = False

        # Limit the number of points sampled from an outline
        self.train_max_num_points = 512
        self.eval_max_num_points = 1024

        # Limit the minimal distance between points sampled from
        # an outline during the training/evaluation process.
        self.sample_by_resolution = True
        self.min_train_sample_point_dist = 2
        self.min_eval_sample_point_dist = 1
        self.eval_upsample = True

        # The distance along the outline between points to use for
        # computing angles along the outline
        self.angle_sample_dist = 6
        self.force_angle_sample = False

        # The default angle value assigned when angle can't reliably be computed
        # or when angle information is intentionally dropped
        self.angle_default = np.pi

        # The scaling of angle information to the angle representation
        # (sine and cosine) multiplied by two (radius is half of this).
        # This will be multiplied by the global scale.
        self.angle_range = 10

        # Should outlines be normalized, to have a unit radius?
        # WARNING: THIS WILL CAUSE SCALE INFORMATION TO BE LOST. YOU DO NOT!(!!)
        # WANT TO DO THIS EXCEPT FOR PROVING THAT THIS IS A BAD IDEA.
        self.force_unit_radius = False

    @property
    def scaled_max_noise_range(self):
        return self.global_scale * self.max_noise_range

    @property
    def scaled_max_noise_radius(self):
        return self.scaled_max_noise_range / 2

    @property
    def scaled_min_train_sample_point_dist(self):
        return self.global_scale * self.min_train_sample_point_dist

    @property
    def scaled_min_eval_sample_point_dist(self):
        return self.global_scale * self.min_eval_sample_point_dist

    @property
    def train_angle_sample_radius(self):
        return int(round(
            self.angle_sample_dist / self.min_train_sample_point_dist
        ))

    @property
    def eval_angle_sample_radius(self):
        return int(round(
            self.angle_sample_dist / self.min_eval_sample_point_dist
        ))

    @property
    def scaled_angle_rep_radius(self):
        return self.global_scale * self.angle_range / 2


class ModelSpec(NestedDict):
    def __init__(self, n_classes):
        self.discard_groups = False
        self.use_angles = True
        self.angle_keep_prob = 0.7
        self.group_hot = True
        self.group_one_hot = False
        self.separate_channels = True
        self.channel_feats = (64, 128, 128, 256,)
        self.channel_neighbors = None
        self.mix_feats = (512, 1024,)
        self.n_classes = n_classes
        self.fc_feats = (512, 256,)
        self.fc_keep_prob = 0.7
        self.pointcnn = False
        self.outline_net = True


class OutlineNetConfig(NestedDict):
    overrides = []

    @staticmethod
    def set_overrides(overrides):
        OutlineNetConfig.overrides.extend(overrides)

    def __init__(self, n_classes=None):
        self.learning_spec = LearningSpec()
        self.data_spec = DataSpec()
        self.model_spec = ModelSpec(n_classes=n_classes)
        self.pointcnn_spec = PointCNNSpec(num_class=n_classes)
        self.use_images = False
        for name, val in OutlineNetConfig.overrides:
            self.set_via_string(name, val)

    def set_n_classes(self, val):
        self.model_spec.n_classes = val
        self.pointcnn_spec.num_class = val

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, val in OutlineNetConfig.overrides:
            self.set_via_string(name, val)


def multi_point_conv(input_pcl, is_training, name, n_feats, n_neighbors=None,
                     activation=tf.nn.relu, debug=None):
    """
    Compute a single "channel" of OutlineNet, by repeatedly applying a per-point
    convolution on the given input pointcloud.
    :param input_pcl: The input pointcloud
    :param is_training: A node indicating whether we are in training mode
    :param name: The name scope for the computation
    :param n_feats: An array specifying the number of point features at each
        layer of the convolution.
    :param activation: The activation function to apply after each convolution.
    :param debug: A dictionary to store direct access to each layer, optional.
    :return: The resulting point multi-layered convolution.
    """
    assert len(input_pcl.shape) in (3, 4)
    assert input_pcl.shape[-1] is not None
    if n_neighbors is None:
        n_neighbors = [1] * len(n_feats)
    else:
        assert len(n_neighbors) == len(n_feats)
    with tf.variable_scope(name):
        if len(input_pcl.shape) == 3:
            # [batch, pt, 1, n_feat]
            input_image = tf.expand_dims(input_pcl, -2)
        else:
            input_image = input_pcl
        net = input_image
        for i, (n_feat, n_neigh) in enumerate(zip(n_feats, n_neighbors)):
            curr_name = 'conv%d' % (i + 1)
            net = tf_util.conv2d(net, n_feat, [n_neigh, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope=curr_name, activation_fn=activation)
            if debug is not None:
                debug[name + '/' + curr_name] = net
        return net


def multi_channel_conv(input_pcls, is_training, n_feats, n_neighbors=None,
                       activation=tf.nn.relu, debug=None):
    """
    Compute multiple "channels" of OutlineNet and concatenate their features.
    :param input_pcls: An ordered list of (name, input) tuples
    :param is_training: A node indicating whether we are in training mode
    :param n_feats: An array specifying the number of point features at each
        layer of the convolution.
    :param activation: The activation function to apply after each convolution.
    :param debug: A dictionary to store direct access to each layer, optional.
    :return: The concatenated features from all convolutions.
    """
    # Make sure it's ordered, to avoid silly programming issues
    assert isinstance(input_pcls, (list, tuple))
    return tf.concat(
        [multi_point_conv(input_pcl=named_input, is_training=is_training,
                          n_feats=n_feats, n_neighbors=n_neighbors,
                          name=name, activation=activation, debug=debug)
         for name, named_input in input_pcls],
        axis=-1,
        name='channel_concat'
    )


def make_fc(pcl_image, is_training, n_feats, n_classes, name='fc',
            keep_prob=None, activation=tf.nn.relu, debug=None):
    """
    Construct the features and logits of the network via fully connected layers.
    :param pcl_image: A 4D pointcloud image
    :param is_training: A node indicating whether we are in training mode
    :param n_feats: An array specifying the number of point features at each
        fully-connected layer.
    :param n_classes: The number of output classes.
    :param keep_prob: The keep probability for the dropout after each fully
        connected layer.
    :param activation: The activation function to apply after each convolution.
    :param debug: A dictionary to store direct access to each layer, optional.
    :return: The features vector, and the logits vectors.
    """
    assert len(pcl_image.shape) == 4
    net = pcl_image
    net = tf.squeeze(net, axis=2, name='squeeze_pcl')
    net = tf.reduce_max(net, axis=1, name='pcl_reduce_max')

    with tf.variable_scope(name):
        for i, n_feat in enumerate(n_feats):
            curr_name = 'fc%d' % (i + 1)
            net = tf_util.fully_connected(
                net, n_feat, bn=False, is_training=is_training,
                scope=curr_name, activation_fn=activation)
            if debug is not None:
                debug[curr_name] = net

            if keep_prob is not None and keep_prob < 1:
                curr_name = curr_name + '_drop'
                net = tf_util.dropout(net, keep_prob=keep_prob,
                                      is_training=is_training,
                                      scope=curr_name)
                if debug is not None:
                    debug[curr_name] = net

        features = net
        logits = tf_util.fully_connected(net, n_classes, activation_fn=None,
                                         scope='logits')
        if debug is not None:
            debug['logits'] = logits

    return features, logits


def tf_side_group_hot(vals, inside_map):
    return tf_group_hot(
        vals, inside_map,
        [Outline2.INSIDE, Outline2.OUTSIDE]
    )


class OutlineNet:
    def __init__(self, config: OutlineNetConfig):
        self.config = config

    def tf_noise_points(self, pts):
        assert len(pts.shape) == 3
        noise = (
            self.config.data_spec.scaled_max_noise_radius *
            tf.random_uniform(
                shape=tf.shape(pts), minval=-1, maxval=+1,
                dtype=pts.dtype, name='point_jitter'
            )
        )
        return pts + noise

    def tf_random_transform_matrix(self, dims=()):
        with tf.name_scope('random_transform_matrix'):
            angles = tf.random_normal(
                mean=self.config.data_spec.rot_center,
                stddev=self.config.data_spec.rot_std,
                shape=dims,
                name='random_angles'
            )

            return tf_rotation_matrix(angles)

    def tf_random_transform_points(self, batch_points):
        with tf.name_scope('random_transform_points'):
            # [batch, n_pts, n_feat]
            assert len(batch_points.shape) == 3
            batch_size = tf.shape(batch_points)[0]
            random_transform = self.tf_random_transform_matrix((batch_size,))
            return tf_transform_points(batch_points, random_transform)

    def augment_train_points(self, batch_points, is_training):
        if not self.config.data_spec.augment_train:
            return batch_points
        def augment():
            pts = batch_points
            pts = self.tf_random_transform_points(pts)
            # TODO: Distort
            assert not self.config.data_spec.attractor_distort
            pts = self.tf_noise_points(pts)
            return pts
        return tf.cond(is_training, augment, lambda: batch_points)

    def make_lr(self, batch_num):
        if self.config.model_spec.pointcnn:
            return self.config.pointcnn_spec.make_lr(batch_num)
        else:
            return self.config.learning_spec.make_lr(batch_num)

    def get_reg_loss(self):
        if self.config.model_spec.pointcnn:
            mult = self.config.pointcnn_spec.weight_decay
        else:
            mult = 1
        return mult * tf.losses.get_regularization_loss()

    def get_optimizer(self, lr):
        if self.config.model_spec.pointcnn:
            eps = self.config.pointcnn_spec.adam_epsilon
            return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
        else:
            return tf.train.AdamOptimizer(learning_rate=lr)

    def make_losses(self, batch_labels, logits):
        if self.config.model_spec.pointcnn:
            labels_2d = tf.expand_dims(
                batch_labels, axis=-1,
                name='labels_2d'
            )
            labels_tile = tf.tile(
                labels_2d, (1, tf.shape(logits)[1]),
                name='labels_tile'
            )
            batch_labels = labels_tile
        smartloss = c3d.smartloss.SmartLoss(
            n_classes=self.config.model_spec.n_classes,
            logits=logits, labels=batch_labels,
            alpha_input=self.config.learning_spec.smartloss_input_alpha,
            alpha_output=self.config.learning_spec.smartloss_output_alpha,
            gamma=self.config.learning_spec.smartloss_gamma,
            fake=not self.config.learning_spec.use_smartloss,
            out_always=self.config.learning_spec.smartloss_out_always,
            use_fp=self.config.learning_spec.smartloss_fp,
            hit_k=self.config.learning_spec.smartloss_hit_k,
            input_k=self.config.learning_spec.smartloss_input_k,
            output_k=self.config.learning_spec.smartloss_output_k,
            use_in=self.config.learning_spec.smartloss_use_in,
            use_out=self.config.learning_spec.smartloss_use_out,
            out_shift=self.config.learning_spec.smartloss_out_shift,
            normalize=self.config.learning_spec.smartloss_normalize,
            focal_loss=self.config.learning_spec.focal_loss,
            focal_gamma=self.config.learning_spec.focal_loss_gamma,
        )
        classify_loss = smartloss.make_loss()
        return smartloss, classify_loss

    def build(self, batch_points, batch_counts, batch_groups, is_training, debug=None):
        assert len(batch_points.shape) == 3
        assert len(batch_groups.shape) == 2

        with tf.name_scope('inputs'):
            batch_points = tf.identity(batch_points, name='batch_points')
            batch_counts = tf.identity(batch_counts, name='batch_counts')
            batch_groups = tf.identity(batch_groups, name='batch_groups')

        if self.config.model_spec.use_angles and not self.config.data_spec.sample_by_resolution and not self.config.data_spec.force_angle_sample:
            raise AssertionError('Angles do not make sense with fixed point counts')

        with tf.control_dependencies([
            tf.assert_equal(
                tf.shape(batch_points)[1],
                tf.cond(is_training,
                        lambda: self.config.data_spec.train_max_num_points,
                        lambda: self.config.data_spec.eval_max_num_points,
                        )
            )
        ]):
            batch_angles = points_angle(
                points=batch_points,
                counts=batch_counts,
                radius=tf.cond(
                    is_training,
                    lambda: self.config.data_spec.train_angle_sample_radius,
                    lambda: self.config.data_spec.eval_angle_sample_radius,
                ),
                avg_point_distance=(
                    None if not self.config.data_spec.sample_by_resolution
                    else tf.cond(
                        is_training,
                        lambda: self.config.data_spec.scaled_min_train_sample_point_dist,
                        lambda: self.config.data_spec.scaled_min_eval_sample_point_dist,
                    )
                ),
                is_training=is_training,
                keep_prob=self.config.model_spec.angle_keep_prob
            )

        with tf.name_scope('angle_rep'):
            batch_angles = self.config.data_spec.scaled_angle_rep_radius * tf.stack(
                (tf.sin(batch_angles), tf.cos(batch_angles)),
                axis=-1
            )

        assert 1 == len([
            v for v in (
                self.config.model_spec.discard_groups,
                self.config.model_spec.group_hot,
                self.config.model_spec.group_one_hot,
            ) if v
        ])

        with tf.name_scope('group_encoding'):
            shape = tf.shape(batch_points)
            group_one_hot = tf_side_group_hot(
                tf.ones((shape[0], shape[1], 1), dtype=batch_points.dtype),
                batch_groups
            )

        with tf.name_scope('input_encoding'):
            if self.config.model_spec.discard_groups:
                encoded_points = batch_points
                encoded_angles = batch_angles
            elif self.config.model_spec.group_one_hot:
                encoded_points = tf.concat((batch_points, group_one_hot), axis=-1)
                encoded_angles = tf.concat((batch_angles, group_one_hot), axis=-1)
            elif self.config.model_spec.group_hot:
                encoded_points = tf_side_group_hot(batch_points, batch_groups)
                encoded_angles = tf_side_group_hot(batch_angles, batch_groups)
            else:
                raise AssertionError()

            assert self.config.model_spec.pointcnn ^ self.config.model_spec.outline_net

            if self.config.model_spec.pointcnn:
                # For PointCNN we will never modify the points - we only modify the
                # angles with the side information.
                batch_angles = encoded_angles
            elif self.config.model_spec.outline_net:
                # For our own network, we can add side information to both the
                # points and the angles. However, if we don't use separate channels
                # for angles and points, it's pointless, so we'll modify only the
                # points (or angles, doesn't matter).
                batch_points = encoded_points
                if self.config.model_spec.separate_channels or not self.config.model_spec.group_one_hot:
                    batch_angles = encoded_angles
            else:
                raise AssertionError()

            batch_points = tf.identity(batch_points, name='batch_points')
            batch_angles = tf.identity(batch_angles, name='batch_angles')

        if self.config.model_spec.pointcnn:
            with tf.name_scope('pointcnn_model'):
                assert self.config.data_spec.sample_by_resolution is False
                assert self.config.data_spec.train_max_num_points == self.config.data_spec.eval_max_num_points
                if not self.config.pointcnn_spec.use_extra_features:
                    features = None
                elif self.config.model_spec.use_angles:
                    assert self.config.data_spec.force_angle_sample
                    features = batch_angles
                else:
                    features = group_one_hot
                import pointcnn_cls
                point_cnn = pointcnn_cls.Net(
                    points=batch_points, features=features,
                    is_training=is_training, setting=self.config.pointcnn_spec,
                    reshape_features=False
                )
                features = point_cnn.fc_layers[-1]
                logits = point_cnn.logits
                flat_logits = tf.reduce_mean(logits, axis=1, name='flat_logits')
        elif self.config.model_spec.outline_net:
            with tf.name_scope('outline_net_model'):
                if not self.config.model_spec.use_angles:
                    pre_combine = multi_point_conv(
                        input_pcl=batch_points, is_training=is_training,
                        n_feats=self.config.model_spec.channel_feats,
                        n_neighbors=self.config.model_spec.channel_neighbors,
                        name='points', debug=debug,
                    )
                elif not self.config.model_spec.separate_channels:
                    pre_combine = multi_point_conv(
                        input_pcl=tf.concat((batch_points, batch_angles), axis=-1),
                        is_training=is_training,
                        n_feats=self.config.model_spec.channel_feats,
                        n_neighbors=self.config.model_spec.channel_neighbors,
                        name='points_and_angles', debug=debug
                    )
                else:
                    pre_combine = multi_channel_conv(
                        input_pcls=(('points', batch_points),
                                    ('angles', batch_angles)),
                        is_training=is_training,
                        n_feats=self.config.model_spec.channel_feats,
                        n_neighbors=self.config.model_spec.channel_neighbors,
                        debug=debug
                    )

                combine = multi_point_conv(
                    input_pcl=pre_combine, is_training=is_training,
                    n_feats=self.config.model_spec.mix_feats,
                    name='combined', debug=debug
                )

                features, logits = make_fc(
                    pcl_image=combine, is_training=is_training,
                    n_feats=self.config.model_spec.fc_feats,
                    n_classes=self.config.model_spec.n_classes,
                    keep_prob=self.config.model_spec.fc_keep_prob,
                    debug=debug
                )
                flat_logits = logits
        else:
            raise AssertionError()

        return features, logits, flat_logits
