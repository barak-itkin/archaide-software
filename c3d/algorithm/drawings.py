import numpy as np
from c3d.datamodel.outline import Outline
from c3d.datamodel.outline2 import Outline2
from c3d.datamodel.profile import Profile
from c3d.datamodel.profile2 import Profile2


# The outer profile of the base.
# Typically appears in GREEN in the drawings.
BASE_OUTER_PROFILE_ID = 'outer_base_profile'


# The inner profile of the base.
# Typically appears in RED in the drawings.
BASE_INNER_PROFILE_ID = 'inner_base_profile'


# The outer profile of the handle.
# Typically appears in YELLOW in the drawings.
HANDLE_OUTER_PROFILE_ID = 'outer_handle_profile'


# The inner profile of the handle.
# Typically appears in BLUE in the drawings.
HANDLE_INNER_PROFILE_ID = 'inner_handle_profile'


# The cut section of of the handle.
# Typically appears in CYAN in the drawings.
HANDLE_CUT_SECTION_ID = 'handle_section'


# The rotation axis for the base profile.
# Typically appears in CYAN in the drawings.
ROTATION_AXIS_ID = 'axis'


# An explicitly annotated fracture segment denoting an area with partial
# data that was completed in the drawing, and not representing the actual
# features of the class.
# Typically appears in MAGENTA in the drawings.
FRACTURE_IDS = {'fracture', 'fracture_1'}


# IDs of all known parts in the SVG document
PART_IDs = {
    BASE_OUTER_PROFILE_ID,
    BASE_INNER_PROFILE_ID,
    HANDLE_OUTER_PROFILE_ID,
    HANDLE_INNER_PROFILE_ID,
    HANDLE_CUT_SECTION_ID,
} | FRACTURE_IDS


class DrawingError(ValueError):
    pass


def distance_squared(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def compute_profile(drawing, should_meet=True, has_rot_axis=True):
    assert BASE_INNER_PROFILE_ID in drawing
    assert BASE_OUTER_PROFILE_ID in drawing
    if has_rot_axis and ROTATION_AXIS_ID not in drawing:
        raise DrawingError('Missing rotation axis')

    if FRACTURE_IDS & drawing.keys():
        raise DrawingError('Not yet supported - drawings with fracture areas')

    inner = list(drawing[BASE_INNER_PROFILE_ID].points)
    outer = list(drawing[BASE_OUTER_PROFILE_ID].points)

    if should_meet:
        # When the inner and outter profiles meet, it should be on top most
        # point, which is known as the rim point.
        # We want the inner to go bottom-up, there meet the outer that goes from
        # top to bottom. Also, note that THE Y AXIS IS GOING DOWN!

        # Is the last point in inner lower (higher Y) than the first?
        if inner[-1][1] > inner[0][1]:
            inner.reverse()

        # Is the last point in outer higher (lower Y) than the first?
        if outer[-1][1] < outer[0][1]:
            outer.reverse()

        # After finishing this, inner should finish where outer begins
        if not inner[-1] == outer[0]:
            raise DrawingError('Inner and outer edges do not meet!')

        result = inner + outer[1:]
    else:
        # Compute the distances if closing the paths by inner + outter
        d1, d2 = distance_squared(inner[-1], outer[0]), distance_squared(outer[-1], inner[0])
        # Compute the distances if closing the paths by inner + reversed(outter)
        d1_alt, d2_alt = distance_squared(inner[-1], outer[-1]), distance_squared(outer[0], inner[0])
        # Check if we get smaller distances when reversing the outter
        if d1 + d2 > d1_alt + d2_alt:
            outer.reverse()

        result = inner + outer

    if has_rot_axis:
        axis = list(drawing[ROTATION_AXIS_ID].points)
        axis_x = axis[0][0]
        result = [(x - axis_x, y) for (x, y) in result]

        if not (result[0][0] == result[-1][0] == 0):
            raise DrawingError("Inner and outer profiles don't start on the same vertical axis!")

    # Now flip the Y axis so that it goes up, which makes more sense in 3D later.
    result = [(p[0], -p[1]) for p in result]
    # Finally return the result
    return Profile(outline=Outline(result), switch_index=len(inner))


def merge_outlines(outline2_a, outline2_b, distance_sq_th):
    assert isinstance(outline2_a, Outline2)
    assert len(outline2_a) > 0
    assert isinstance(outline2_b, Outline2)
    assert len(outline2_b) > 0
    assert distance_squared(outline2_a[-1], outline2_b[0]) < distance_sq_th

    return Outline2(
        points=outline2_a.points[:-1] + outline2_b.points,
        inside_map=outline2_a.inside_map + outline2_b.inside_map
    )


def _merge_distances(outline2_a, outline2_b):
    assert isinstance(outline2_a, Outline2)
    assert outline2_a.is_open
    assert isinstance(outline2_b, Outline2)
    assert outline2_a.is_open

    d1 = distance_squared(outline2_a[0], outline2_b[0])
    d2 = distance_squared(outline2_a[0], outline2_b[-1])
    d3 = distance_squared(outline2_a[-1], outline2_b[0])
    d4 = distance_squared(outline2_a[-1], outline2_b[-1])

    return d1, d2, d3, d4


def min_merge_distance(outline2_a, outline2_b):
    return min(_merge_distances(outline2_a, outline2_b))


def try_combine_outlines(outline2_a, outline2_b, distance_th):
    d1, d2, d3, d4 = _merge_distances(outline2_a, outline2_b)

    min_dist = min(d1, d2, d3, d4)
    if min_dist > distance_th:
        return None

    if d1 == min_dist:
        return merge_outlines(outline2_b.reversed(), outline2_a, distance_th)
    elif d2 == min_dist:
        return merge_outlines(outline2_b, outline2_a, distance_th)
    elif d3 == min_dist:
        return merge_outlines(outline2_a, outline2_b, distance_th)
    elif d4 == min_dist:
        return merge_outlines(outline2_a, outline2_b.reversed(), distance_th)
    else:
        raise AssertionError()


def try_close(outline2, distance_th):
    if len(outline2) <= 2 or outline2.is_closed:
        return outline2
    elif distance_squared(outline2[0], outline2[-1]) <= distance_th:
        return Outline2(
            points=outline2.points[:-1],
            inside_map=outline2.inside_map
        )
    else:
        return outline2


def compute_profile2(drawing, distance_th=2, has_rot_axis=True,
                     regular_y=False, should_close=None, expect_single=True,
                     assert_full=True):
    assert BASE_INNER_PROFILE_ID in drawing or not assert_full
    assert BASE_OUTER_PROFILE_ID in drawing or not assert_full
    if has_rot_axis and ROTATION_AXIS_ID not in drawing:
        raise DrawingError('Missing rotation axis')

    outlines = []
    # To gaurantee consistent output, always sort the outlines by name
    for name, old_outline in sorted(drawing.outlines.items()):
        assert isinstance(old_outline, Outline)
        if 'handle' in name or name == ROTATION_AXIS_ID:
            continue
        elif 'fracture' in name:
            o_type = Outline2.NEITHER
        elif 'inner_base' in name:
            o_type = Outline2.INSIDE
        elif 'outer_base' in name:
            o_type = Outline2.OUTSIDE
        else:
            raise DrawingError('Unknown segment %s' % name)
        outlines.append(
            Outline2(points=old_outline.points,
                     inside_map=[o_type] * (len(old_outline) - 1))
        )

    # Try merging all pairs of outlines that can be merged
    while len(outlines) > 1:
        n = len(outlines)

        # Find the pair of outlines with the lowest merge distance
        i, j = min(
            ((i, j) for i in range(n) for j in range(i + 1, n)),
            key=lambda pair: min_merge_distance(outlines[pair[0]], outlines[pair[1]])
        )

        outline_a = outlines[i]
        outline_b = outlines[j]
        combined = try_combine_outlines(outline_a, outline_b, distance_th)

        if combined:
            # Delete larger index first
            del outlines[j]
            del outlines[i]
            outlines.append(combined)
        else:
            break

    for i, o in enumerate(outlines):
        outlines[i] = try_close(o, distance_th)

    if expect_single and len(outlines) > 1:
        raise DrawingError('Failed to merge drawing to a single outline!')

    if has_rot_axis:
        axis_x = drawing[ROTATION_AXIS_ID].points[0][0]
    else:
        axis_x = 0

    y_mul = +1 if regular_y else -1

    # Now:
    # 1. Align all outlines so that rotation axis is at X = 0.
    # 2. Try close every outline if it's already a tight circle
    outlines = [
        try_close(
            o.transform(lambda p: (p[0] - axis_x, y_mul * p[1])),
            distance_th
        )
        for o in outlines
    ]

    any_closed = any(o.is_closed for o in outlines)
    if any_closed and should_close is False:
        raise DrawingError('Merge result is closed form, but expected an open form!')

    all_closed = all(o.is_closed for o in outlines)
    if not all_closed and should_close is True:
        raise DrawingError('Merge result is open form, but expected a closed form!')

    # Finally return the result
    return Profile2(outlines=outlines)


def make_cw(outline2):
    # Check the orientation of the points using the shoelace formula
    # https://stackoverflow.com/a/1165943/748102
    # https://en.wikipedia.org/wiki/Shoelace_formula
    pts = np.asanyarray(outline2.points)
    pts_prev = np.roll(pts, -1, axis=0)
    area = np.nansum(
        (pts[:, 0] - pts_prev[:, 0]) * (pts[:, 1] + pts_prev[:, 1]),
        axis=0
    )
    assert not np.isnan(area)  # Make sure we have valid points!
    if area > 0:
        return outline2
    else:
        return outline2.reversed()


def _rotate_first(arr, first_idx):
    assert isinstance(arr, list)
    assert first_idx >= 0
    return arr[first_idx:] + arr[:first_idx]


def make_consistent(outline2):
    if not outline2.is_closed:
        return outline2
    min_pt_idx = min(range(len(outline2)), key=lambda i: outline2.points[i])
    return Outline2(
        points=_rotate_first(outline2.points, min_pt_idx),
        inside_map=_rotate_first(outline2.inside_map, min_pt_idx)
    )
