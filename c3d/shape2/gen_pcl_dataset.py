import c3d.algorithm.fracture
import c3d.algorithm.drawings
import c3d.algorithm.sample as sample_m
import c3d.classification.dataset
import c3d.datamodel
import c3d.shape2.input
import c3d.pipeline.rasterizer
import numpy as np

from c3d.datamodel import Profile2, Outline2
from c3d.shape2 import outlinenet


BLACKLIST = [
    ('CF_22', 'GR007_001_profile.svg'),
    ('CF_20_2', 'MGB003_004_profile.svg'),
    ('CF_26_1', 'PG001_001_profile.svg'),
    ('CF_11', 'GR039_002_profile.svg'),
]


class FractureSamplingInputMixin(c3d.classification.dataset.Dataset):
    def __init__(self, *args, **kwargs):
        super(FractureSamplingInputMixin, self).__init__(*args, **kwargs)
        self.data_spec = None  # type: outlinenet.DataSpec
        self.eval_mode = None  # type: bool
        self.insert_invalid_points = False

    def set_config(self, config, eval_mode):
        self.data_spec = config
        self.eval_mode = eval_mode

    @property
    def num_points(self):
        if self.eval_mode and self.data_spec.eval_upsample:
            return self.data_spec.eval_max_num_points
        else:
            return self.data_spec.train_max_num_points

    @property
    def sample_points_by_dist(self):
        if self.eval_mode and self.data_spec.eval_upsample:
            return self.data_spec.scaled_min_eval_sample_point_dist
        else:
            return self.data_spec.scaled_min_train_sample_point_dist

    def file_id_filter(self, file_id):
        if (file_id.label, file_id.id) in BLACKLIST or (file_id.source_label, file_id.id) in BLACKLIST:
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

    def scale_input_outline(self, o, sx, sy):
        pts = np.asanyarray(o.points)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        return Outline2(points=pts, inside_map=o.inside_map)

    def normalize_points(self, pts):
        pts = np.asanyarray(pts)
        assert pts.ndim == 2 and pts.shape[1] == 2
        center = np.nanmean(pts, axis=0)
        result = pts - center
        if self.data_spec.force_unit_radius:
            result /= np.max(np.linalg.norm(result, axis=1))
        return result

    def _pad_extra_point_values(self, point_vals):
        if len(point_vals) < self.num_points:
            n = len(point_vals)
            new_shape = list(point_vals.shape)
            new_shape[0] = self.num_points
            result = np.zeros(new_shape, dtype=point_vals.dtype)
            result[:n] = point_vals
            return result
        else:
            return point_vals

    def _pad_extra_map_values(self, map_vals):
        n = len(map_vals)
        if n < self.num_points:
            result_map = np.full((self.num_points,), Outline2.NEITHER)
            result_map[:n] = map_vals
            return result_map
        else:
            return map_vals

    def _insert_invalid_points(self, points, inside_map, pad_size=1):
        n_points = len(points)
        pair_distances = sample_m.pair_distances(points)
        # Find which segments are actually skipping along the outline
        skip_segments = pair_distances > 2 * self.sample_points_by_dist
        # Insert an invalid point in the middle
        if np.any(skip_segments):
            skip_count = np.concatenate(([0], np.cumsum(skip_segments))) * pad_size
            new_pts = np.zeros((n_points + skip_count[-1], 2))
            new_map = np.full((n_points + skip_count[-1],), Outline2.NEITHER, dtype=np.int32)
            dst_indices = np.arange(n_points) + skip_count
            new_pts[dst_indices] = points
            new_map[dst_indices] = inside_map

            points = new_pts
            inside_map = new_map

            assert len(points) <= self.num_points
        return points, inside_map

    def random_normal(self, mean, std, min_val):
        while True:
            s = np.random.normal(loc=mean, scale=std)
            if s >= min_val:
                return s

    def random_scale(self):
        return self.random_normal(
            mean=self.data_spec.scale_center,
            std=self.data_spec.scale_std,
            min_val=self.data_spec.scale_min
        )

    def random_hscale(self):
        return self.random_normal(
            mean=self.data_spec.hscale_center,
            std=self.data_spec.hscale_std,
            min_val=self.data_spec.hscale_min
        )

    def per_run_augment(self, file_id, prepared):
        outline2 = prepared

        # To get the right number of points, we want to do scaling augmentation
        # before the sampling. However, this process is slow (running on the CPU
        # without parallelization) and runs on thousands of points.

        # Instead, we approximate the new outline length after the scaling,
        # sample the original outline in the scaled resolution, and then only
        # scale the sampled points.

        # This "cheap" heuristic makes the entire pipeline run almost 3x faster
        # as most outlines get sampled for dozens of points, which is
        # significantly less than the thousands of the original outline.

        augment_scale = not self.eval_mode and self.data_spec.augment_train
        if augment_scale:
            scale = self.random_scale()
            hscale = self.random_hscale()
            sy = scale
            sx = hscale * scale
            mix_scale = scale * ((1 + hscale) / 2)  # Heuristic
        else:
            mix_scale = 1

        length = sample_m.get_outline2_length(outline2, skip_invalid_segs=True) * mix_scale

        if not self.data_spec.sample_by_resolution:
            n_points = self.num_points
        else:
            n_points = max(3, int(length // self.sample_points_by_dist))
            # assert n_points <= self.num_points * 1.5
            n_points = min(n_points, self.num_points)

        fractional_distances = sample_m.uniform_fractional_distances(
            n_points, jitter_size=(0 if self.eval_mode else 0.5)
        )
        # Sort, just to make sure nothing went wrong in the process
        fractional_distances.sort()

        points, inside_map = sample_m.sample_outline2_at_distances(
            fractional_distances=fractional_distances,
            outline=outline2,
            distances_sorted=True,
            skip_invalid_segs=True
        )

        # Normalize right before repeating the last point, to make sure we have
        # the correct center.
        points = self.normalize_points(points)

        if augment_scale:
            points[:, 0] *= sx
            points[:, 1] *= sy

        if self.insert_invalid_points:
            points, inside_map = self._insert_invalid_points(points, inside_map, pad_size=1)

        n_points = len(points)
        if n_points < self.num_points:
            points = self._pad_extra_point_values(points)
            inside_map = self._pad_extra_map_values(inside_map)

        return points, inside_map, n_points

    def prepare_file(self, file_id):
        profile2 = super(FractureSamplingInputMixin, self).prepare_file(file_id)
        try:
            assert isinstance(profile2, Profile2)
            assert len(profile2.outlines) == 1
            outline2 = self.scale_input_outline(
                profile2.outlines[0],
                sx=self.data_spec.global_scale,
                sy=self.data_spec.global_scale,
            )
            # IMPORTANT: If we want to get angles consistent, all outlines must be
            # oriented in the same direction!
            outline2 = c3d.algorithm.drawings.make_cw(outline2)
            outline2 = c3d.algorithm.drawings.make_consistent(outline2)
            return outline2
        except:
            print('Error preparing %s' % str(file_id))
            raise


class FractureImageInputMixin(FractureSamplingInputMixin):
    class PointsHandler:
        def get_outline(self, points):
            return c3d.datamodel.Outline(points)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rasterizer = c3d.pipeline.rasterizer.Rasterizer(
            self.PointsHandler(),
            width=256, height=256, antialias=False
        )

    def per_run_augment(self, file_id, prepared):
        points, inside_map, n_points = super().per_run_augment(
            file_id, prepared
        )
        return np.asarray(self.rasterizer.rasterize(points))


class ProfileDataset(FractureSamplingInputMixin, c3d.shape2.input.ProfileDataset):
    pass


class SherdDataset(FractureSamplingInputMixin, c3d.shape2.input.ProfileFractureDataset):
    pass


class SherdSVGDataset(FractureSamplingInputMixin, c3d.shape2.input.SherdSVGDataset):
    pass


class ProfileImageDataset(FractureImageInputMixin, c3d.shape2.input.ProfileDataset):
    pass


class SherdImageDataset(FractureImageInputMixin, c3d.shape2.input.ProfileFractureDataset):
    pass


class SherdSVGImageDataset(FractureImageInputMixin, c3d.shape2.input.SherdSVGDataset):
    pass
