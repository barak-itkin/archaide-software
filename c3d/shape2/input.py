from c3d.algorithm.drawings import compute_profile2
from c3d.classification import Dataset
from c3d.datamodel import load_json, Profile, Profile2, Outline2
from c3d.importer import svg_import


class ProfileDataset(Dataset):
    def file_filter(self, dirname, filename):
        return filename.endswith('.profile.json')

    def prepare_file(self, file_id):
        profile = load_json(self.file_path(file_id), Profile)
        points = profile.outline.points
        inside_map = (
            [Outline2.INSIDE] * profile.switch_index +
            [Outline2.OUTSIDE] * (len(profile.outline) - profile.switch_index - 1)
        )
        profile2 = Profile2(
            outlines=[Outline2(points=points, inside_map=inside_map)],
            simplification_area=profile.simplification_area
        )
        return profile2


class ProfileFractureDataset(Dataset):
    def file_filter(self, dirname, filename):
        return filename.endswith('.fracture-prof2.json')

    def prepare_file(self, file_id):
        return load_json(self.file_path(file_id), Profile2)


class SherdSVGDataset(Dataset):
    def __init__(self, *args, regular_y=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.regular_y = regular_y

    def file_filter(self, dirname, filename):
        return filename.endswith('.svg')

    def prepare_file(self, file_id):
        drawing = svg_import.drawing_from_svg(self.file_path(file_id))
        profile2 = compute_profile2(drawing, has_rot_axis=False,
                                    regular_y=self.regular_y)
        return profile2


class ProfileSVGDataset(Dataset):
    def __init__(self, *args, regular_y=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.regular_y = regular_y

    def file_filter(self, dirname, filename):
        return filename.endswith('.svg')

    def prepare_file(self, file_id):
        drawing = svg_import.drawing_from_svg(self.file_path(file_id))
        profile2 = compute_profile2(drawing, has_rot_axis=True,
                                    regular_y=self.regular_y)
        return profile2
