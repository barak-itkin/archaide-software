import c3d.datamodel
import numpy as np

from c3d.datamodel import Outline2


def pair_distances(polyline, segments_to_zero=None):
    """
    Given a polyline (N x 2D points), return the distances between every two
    consecutive points.
    """
    polyline = np.asanyarray(polyline)
    d_vec = polyline[1:, :] - polyline[:-1, :]
    result = np.linalg.norm(d_vec, axis=1)
    if segments_to_zero is not None:
        result[segments_to_zero] = 0
    return result


def distances_from_start(polyline, segments_to_zero=None):
    """
    Given a polyline (N x 2D points), return the cumulative distances between
    points and the start point.
    """
    return np.cumsum(pair_distances(polyline,
                                    segments_to_zero=segments_to_zero))


def outline_length(points, segments_to_zero=None):
    """
    Given a polyline (N x 2D points), return the total length of the polyline.
    """
    return np.sum(pair_distances(points,
                                 segments_to_zero=segments_to_zero))


def sample_at_distances(polyline, fractional_distances,
                        distances_sorted=False, segments_to_zero=None,
                        distances_are_absolute=False, eps=1e-5):
    """
    Sample points along the given poly-line, using the given fractional
    distances.
    """
    polyline = np.array(polyline)  # Avoid modifying the inputs
    fractional_distances = np.array(fractional_distances)  # Avoid modifying the inputs

    if not distances_sorted:
        fractional_distances.sort()

    d = distances_from_start(polyline, segments_to_zero)
    total_distance = d[-1]
    assert total_distance > 0

    fractional_d = d / total_distance
    assert fractional_d[-1] == 1

    if distances_are_absolute:
        fractional_distances /= total_distance

    if any(d < -eps or d > 1 + eps for d in fractional_distances):
        raise ValueError('Invalid fractional distances outside of range [0, 1]!')

    seg_index = 0
    dist_passed = 0
    result = np.zeros((len(fractional_distances), 2))
    segments = np.zeros(len(fractional_distances), dtype=int)
    for i, dist in enumerate(fractional_distances):
        # Note that to avoid numeric issues that might cause us to pass by epsilon the
        # last segment, we add an explicit check.
        while seg_index < len(fractional_d) - 1 and (
            fractional_d[seg_index] < dist or (
                # This check is needed in case the first point is as 0 and the
                # first segment is a segment to skip.
                segments_to_zero is not None and segments_to_zero[seg_index]
            )
        ):
            dist_passed = fractional_d[seg_index]
            seg_index += 1

        # Assume we have a segment between a and b, and we want to sample point
        # at distance c (where a <= c <= b), then the fraction of the segment is
        # (c - a) / (b - a)
        fraction = (dist - dist_passed) / (fractional_d[seg_index] - dist_passed)
        result[i] = (1 - fraction) * polyline[seg_index] + fraction * polyline[seg_index + 1]
        segments[i] = seg_index

    return result, segments


def angle_between(vec1s, vec2s):
    x1, y1 = vec1s[:, 0], vec1s[:, 1]
    x2, y2 = vec2s[:, 0], vec2s[:, 1]
    # See https://stackoverflow.com/a/16544330
    dot = x1 * x2 + y1 * y2  # dot product between [x1, y1] and [x2, y2]
    det = x1 * y2 - y1 * x2  # determinant
    return np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)


def angles_at_distances(polyline, distances, sample_dist,
                        distances_sorted=False, segments_to_zero=None,
                        cyclic=False):
    """
    Sample points along the given poly-line, using the given fractional
    distances.
    """
    length = outline_length(polyline, segments_to_zero)

    if not distances_sorted:
        distances.sort()

    if cyclic:
        d_start = (distances - sample_dist) % length
        d_end = (distances + sample_dist) % length
    else:
        d_start = np.maximum(distances - sample_dist, 0)
        d_end = np.minimum(distances + sample_dist, length)

    prevs, prev_segs = sample_at_distances(
        polyline, d_start, distances_sorted=True, segments_to_zero=segments_to_zero,
        distances_are_absolute=True
    )
    pts, pt_segs = sample_at_distances(
        polyline, distances, distances_sorted=True, segments_to_zero=segments_to_zero,
        distances_are_absolute=True
    )
    nexts, next_segs = sample_at_distances(
        polyline, d_end, distances_sorted=True, segments_to_zero=segments_to_zero,
        distances_are_absolute=True
    )

    to_prev = prevs - pts
    to_prev_norm = np.linalg.norm(to_prev, axis=-1)
    to_prev_n = to_prev / np.expand_dims(to_prev_norm, axis=-1)
    to_next = nexts - pts
    to_next_norm = np.linalg.norm(to_next, axis=-1)
    to_next_n = to_next / np.expand_dims(to_next_norm, axis=-1)

    result = angle_between(to_prev_n, to_next_n)

    # Technically, we may acheive NaN during the normalization above.
    bad_mask = np.isnan(result)
    # Other problems can be if the sample skipped across "invalid" segments (segments to zero).
    # Detecting this is hard, but we can assume that if these segments are of non-negligible
    # length, then the resulting points would just be farther away than the circle of "sample_dist".
    # TODO: Better clamping to angle computation inside regions
    bad_mask |= (to_prev_norm > sample_dist * 1.01) | (to_next_norm > sample_dist * 1.01)

    # For bad points, set the angle to 180 (i.e. not interesting whatsoever).
    result[bad_mask] = np.pi

    return result


def get_outline2_polyline(outline):
    """
    Return a polyline from an Outline2, repeating the first point at the end
    for closed outlines.
    """
    assert isinstance(outline, c3d.datamodel.Outline2)
    if outline.is_open:
        return np.asarray(outline.points)
    else:
        return np.vstack(
            (outline.points, [outline.points[0]])
        )


def get_outline2_length(outline, skip_invalid_segs=True):
    polyline = get_outline2_polyline(outline)
    segments_to_zero = np.array([
        inside == Outline2.NEITHER for inside in outline.inside_map
    ]) if skip_invalid_segs else None
    return outline_length(polyline, segments_to_zero)


def sample_outline2_at_distances(outline, fractional_distances,
                                 distances_sorted=False,
                                 skip_invalid_segs=True):
    """
    Similar to `sample_at_distances`, but for Outline2 objects. Also return the
    inside/outside/etc. status for each point.
    """
    assert isinstance(outline, c3d.datamodel.Outline2)
    polyline = get_outline2_polyline(outline)
    segments_to_zero = np.array([
        inside == Outline2.NEITHER for inside in outline.inside_map
    ]) if skip_invalid_segs else None
    sampled_points, sample_segments = sample_at_distances(
        polyline, fractional_distances, distances_sorted, segments_to_zero
    )
    sampled_inside_map = [outline.inside_map[s] for s in sample_segments]
    return sampled_points, sampled_inside_map


def sample_outline2_angles_at_distances(outline, distances, sample_dist,
                                               distances_sorted=False,
                                               skip_invalid_segs=True):
    assert isinstance(outline, c3d.datamodel.Outline2)
    polyline = get_outline2_polyline(outline)
    segments_to_zero = np.array([
        inside == Outline2.NEITHER for inside in outline.inside_map
    ]) if skip_invalid_segs else None
    return angles_at_distances(
        polyline, distances=distances, sample_dist=sample_dist, distances_sorted=distances_sorted,
        segments_to_zero=segments_to_zero, cyclic=not outline.is_open
    )


def uniform_fractional_distances(count, jitter_size=0):
    """
    Fractional distances for uniformly sampling `count` points over a polyline.
    """
    if jitter_size <= 0:
        return np.arange(count) / (count - 1.0)
    else:
        vals = np.arange(count) / count
        # Random jitter forward and back, not more than one "segment" to
        # maintain relative uniformity.
        vals += jitter_size * np.random.uniform(low=-1, high=1, size=vals.shape) / (2 * count)
        # Random shift to the start position
        vals += np.random.uniform(low=0, high=1)
        # Cycle all values that passed
        return vals % 1.
