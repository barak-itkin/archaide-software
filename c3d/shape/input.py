from c3d.classification import Dataset
from c3d.datamodel import Profile, Fracture, load_json


class ProfileDataset(Dataset):
    def label_from_path(self, dirpath, filename):
        return filename.split('.')[0]

    def id_from_path(self, dirpath, filename):
        return dirpath

    def file_name(self, file_id):
        return '%s.profile.json' % file_id.label

    def file_dir(self, file_id):
        return file_id.dir

    def file_filter(self, dirname, filename):
        return filename.endswith('.profile.json')

    def prepare_file(self, file_id):
        return load_json(self.file_path(file_id), Profile)


class SherdDataset(Dataset):
    def file_filter(self, dirname, filename):
        return filename.endswith('.fracture.json')

    def id_from_path(self, dirpath, filename):
        # Files are simply numbered
        return int(filename.split('.')[0], 10)

    def file_name(self, file_id):
        return '%05d.fracture.json' % file_id.id

    def prepare_file(self, file_id):
        return load_json(self.file_path(file_id), Fracture)
