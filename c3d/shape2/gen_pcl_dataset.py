import argparse
import c3d.algorithm.fracture
import c3d.algorithm.drawings
import c3d.algorithm.sample as sample_m
import c3d.classification.dataset
import c3d.datamodel
import c3d.shape2.input
import h5py
import numpy as np
import os

from c3d.datamodel import Profile2, Outline2


# Current number of "values" per point
NUM_POINT_DIMENSIONS = 8

# Scale all points from mm to meters
GLOBAL_SCALE = 0.01

# Compute angles by looking 1mm to each side
ANGLE_SAMPLE_DIST = 2

# Add a random scale augmentation
SCALE_CENTER = 1.2
SCALE_STD = 0.8

# Add a random horizontal scale augmentation, simulating fractures that are not
# exactly parallel to the image plane.
HSCALE_CENTER = 1
HSCALE_STD = 0.2

# Add a random rotation, as fractures aren't always exactly straightened up.
ROT_CENTER = 0
ROT_STD = np.deg2rad(10)


def random_scale():
    while True:
        val = np.random.normal(SCALE_CENTER, SCALE_STD)
        if val > 0.8:
            return val


def scale_matrix(s=None):
    s = random_scale() if s is None else s
    return np.asarray([
        [s, 0],
        [0, s],
    ])


def random_hscale():
    return np.random.normal(HSCALE_CENTER, HSCALE_STD)


def hscale_matrix(hs=None):
    hs = random_hscale() if hs is None else hs
    return np.asarray([
        [hs, 0],
        [0,  1],
    ])


def random_rot_in_rad():
    return np.random.normal(ROT_CENTER, ROT_STD)


def rot_matrix(r=None):
    r = random_rot_in_rad() if r is None else r
    sin, cos = np.sin(r), np.cos(r)
    return np.asarray([
        [+cos, -sin],
        [+sin, +cos],
    ])


def random_transform_matrix():
    return np.matmul(
        np.matmul(scale_matrix(), hscale_matrix()),
        rot_matrix()
    )


class FractureSamplingInputMixin(c3d.classification.dataset.Dataset):
    def __init__(self, *args, **kwargs):
        super(FractureSamplingInputMixin, self).__init__(*args, **kwargs)
        self.class_subset = None
        self.num_points = 64
        self.sample_points_by_dist = 3 * GLOBAL_SCALE
        self.max_noise_size = 2 * GLOBAL_SCALE
        self.angle_sampling_distance = ANGLE_SAMPLE_DIST
        self.noise_size = None
        self.do_noise = False

    @property
    def has_class_subset(self):
        return self.class_subset is not None and len(self.class_subset) > 0

    def file_id_filter(self, file_id):
        if self.has_class_subset and file_id.label not in self.class_subset:
            print('Drop subset')
            return False
        if file_id.label != 'CF_15':
            return True
        # TODO(Hack)
        profile2 = super(FractureSamplingInputMixin, self).prepare_file(file_id)
        if len(profile2.outlines) != 1:
            print('Dropping %s on %d outlines' % (str(file_id), len(profile2.outlines)))
            return False
        if all(v == Outline2.NEITHER for v in profile2.outlines[0].inside_map):
            print('Dropping %s on invalid inside map' % (str(file_id)))
            return False

        return True

    def scale_input_outline(self, o):
        return Outline2(
            points=np.asanyarray(o.points) * GLOBAL_SCALE,
            inside_map=o.inside_map
        )

    def normalize_points(self, pts):
        pts = np.asanyarray(pts)
        assert pts.ndim == 2 and pts.shape[1] == 2
        center = np.mean(pts, axis=0)
        return pts - np.tile(np.expand_dims(center, axis=0), (len(pts), 1))

    def noise_points(self, pts):
        if not self.do_noise:
            return pts
        pts = np.asanyarray(pts)
        assert pts.ndim == 2 and pts.shape[1] == 2
        if self.noise_size is None:
            part_length = c3d.algorithm.sample.outline_length(pts) / (len(pts) - 1)
            # noise_size = part_length / 2
            noise_size = max(part_length, self.max_noise_size)
        else:
            noise_size = self.noise_size
        noise = np.random.uniform(low=-1, high=+1, size=pts.shape) / 2 * noise_size
        return pts + noise

    def noise_angles(self, angles):
        if not self.do_noise:
            return angles
        angles = np.asanyarray(angles)
        assert angles.ndim == 1
        noise_size = 0.1 * 2 * np.pi
        noise = np.random.uniform(low=-1, high=+1, size=angles.shape) / 2 * noise_size
        return angles + noise

    def augment_transform_points(self, pts):
        if not self.do_noise:
            return pts
        pts = np.asanyarray(pts)
        assert pts.ndim == 2 and pts.shape[1] == 2
        return np.matmul(
            random_transform_matrix(), pts.T
        ).T

    def add_side_information_and_angle(self, pts, inside_map, angles):
        pts = np.asanyarray(pts)
        assert pts.ndim == 2 and pts.shape[1] == 2
        inside_mask = [v == Outline2.INSIDE for v in inside_map]
        outside_mask = [v == Outline2.OUTSIDE for v in inside_map]

        result = np.zeros((len(pts), 4))
        result[inside_mask, 0:2] = pts[inside_mask]
        result[outside_mask, 2:4] = pts[outside_mask]

        result_angles = np.zeros((len(pts), 4))
        angles_v = 2 * self.max_noise_size / np.pi * np.vstack(
            (np.sin(angles), np.cos(angles))
        ).T
        result_angles[inside_mask, 0:2] = angles_v[inside_mask]
        result_angles[outside_mask, 2:4] = angles_v[outside_mask]

        return np.concatenate((result, result_angles), axis=1)

    def per_run_augment(self, file_id, prepared):
        points, inside_map, angles = prepared
        pts = self.noise_points(points)
        angles = np.abs(angles)
        angles = self.noise_angles(angles)
        pts = self.augment_transform_points(pts)
        pts = self.add_side_information_and_angle(pts, inside_map, angles)
        assert pts.shape[1] == NUM_POINT_DIMENSIONS
        return pts

    def prepare_file(self, file_id):
        profile2 = super(FractureSamplingInputMixin, self).prepare_file(file_id)
        try:
            assert isinstance(profile2, Profile2)
            assert len(profile2.outlines) == 1
            outline2 = self.scale_input_outline(profile2.outlines[0])
            length = sample_m.get_outline2_length(outline2)
            if self.do_noise:
                # Sample roughly every 2mm
                n_points = min(self.num_points, int(length // self.sample_points_by_dist))
                fractional_distances = np.zeros(self.num_points)
                fractional_distances[:n_points] = np.random.uniform(low=0, high=1, size=(n_points,))
                # Repeat the last point as needed
                fractional_distances[n_points:] = fractional_distances[n_points - 1]
            else:
                fractional_distances = sample_m.uniform_fractional_distances(self.num_points)
                # IMPORTANT: This was a uniform sampling. Without transformation on the sherds,
                # this creates an overfit in the angle computation and perhaps more things.
                # fractional_distances = np.random.uniform(n_points)
            fractional_distances.sort()
            distances = fractional_distances * length
            points, inside_map = sample_m.sample_outline2_at_distances(
                fractional_distances=fractional_distances,
                outline=outline2,
                distances_sorted=True,
                skip_invalid_segs=True
            )
            angles = sample_m.sample_outline2_angles_at_distances(
                outline=outline2,
                distances=distances,
                sample_dist=self.angle_sampling_distance,
                distances_sorted=True,
                skip_invalid_segs=True
            )
            assert Outline2.NEITHER not in inside_map
            points = self.normalize_points(points)
            return points, inside_map, angles
        except:
            print('Error preparing %s' % str(file_id))
            raise


class ProfileDataset(FractureSamplingInputMixin, c3d.shape2.input.ProfileDataset):
    pass


class SherdDataset(FractureSamplingInputMixin, c3d.shape2.input.ProfileFractureDataset):
    pass


class SherdSVGDataset(FractureSamplingInputMixin, c3d.shape2.input.SherdSVGDataset):
    pass


INPUT_MAPPING = {
    'profile': ProfileDataset,
    'sherd': SherdDataset,
    'sherd_svg': SherdSVGDataset,
}


def make_parser():
    parser = argparse.ArgumentParser('Generate point cloud datasets')
    parser.add_argument('dest_path', type=str,
                        help='Path to create the .h5 dataset')
    parser.add_argument('data_root', type=str,
                        help='The folder containing the data')
    parser.add_argument('split', choices=['all', 'train', 'test'],
                        help='Which files to generate')
    parser.add_argument('num_points', type=int, default=64,
                        help='Number of points to generate from each file')
    parser.add_argument('--add_noise', default=False, action='store_true',
                        help='Add noise to sampled points')
    parser.add_argument('--input_type', default='sherd', choices=INPUT_MAPPING.keys(),
                        help='Add noise to sampled points')
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    dataset = INPUT_MAPPING[args.input_type](data_root=args.data_root)
    dest = os.path.abspath(args.dest_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    dataset.do_noise = args.add_noise
    dataset.num_points = args.num_points

    if args.split == 'all':
        source = dataset.file_ids_iter()
    elif args.split == 'train':
        source = dataset.file_ids_train_iter()
    elif args.split == 'test':
        source = dataset.file_ids_test_iter()
    else:
        raise AssertionError()

    result = []
    labels = []
    for f_id in source:
        result.append(dataset.prepare_file(f_id))
        labels.append(f_id.label)

    all_labels = list(set(labels))
    label_to_index = dict((v,i) for i,v in enumerate(all_labels))

    result = np.asarray(result)
    label_indices = np.asarray([label_to_index[l] for l in labels])
    all_labels = np.asarray(all_labels, dtype='S')
    with h5py.File(dest, 'w') as f:
        f.create_dataset('data', data=result)
        f.create_dataset('labels', data=label_indices)
        f.create_dataset('index_to_label', data=all_labels)


if __name__ == '__main__':
    main()
