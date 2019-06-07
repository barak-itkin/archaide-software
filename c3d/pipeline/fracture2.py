import argparse
import os
import traceback
import tqdm
from c3d.algorithm.break2 import fracture_profile
from c3d.algorithm.drawings import compute_profile2, DrawingError
from c3d.importer import svg_import
import sys


def load_profile(path, regular_y):
    drawing = svg_import.drawing_from_svg(path)
    return compute_profile2(drawing, regular_y=regular_y)


def little_float(obj):
    print(type(obj))
    if isinstance(obj, float):
        return round(obj)
    else:
        return obj


def generate_fractures(profile2, dst_dir, prefix, start, count, min_length):
    for i in range(start, start + count):
        f_name = '%s_%04d.fracture-prof2.json' % (prefix, i)
        dst_file = os.path.join(dst_dir, f_name)
        if os.path.exists(dst_file):
            continue
        frac = fracture_profile(profile2, min_length).optimize_unused()
        frac.dump(dst_file, default=little_float)


def print_progress(msg):
    print(msg, end='')
    sys.stdout.flush()


def run(src_dir, dst_dir, count, min_length, regular_y):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    jobs = []

    print_progress('Reading profiles')
    for root, dirs, files in os.walk(src_dir):
        rel_src_dir = os.path.relpath(root, src_dir)
        for f_name in files:
            if not f_name.endswith('.svg'):
                continue

            src_path = os.path.join(root, f_name)
            try:
                profile = load_profile(src_path, regular_y)
            except DrawingError:
                print('Error loading %s' % src_path)
                traceback.print_exc()
                continue

            prefix = f_name[:-len('.svg')]
            dst_base = os.path.join(dst_dir, rel_src_dir)
            if not os.path.exists(dst_base):
                os.makedirs(dst_base, exist_ok=True)
            jobs.append((profile, dst_base, prefix))
        print_progress('.')
    print()

    hard_profiles = {}
    with tqdm.tqdm(total=len(jobs) * count) as progress:
        for i in range(count):
            for profile, dst_base, prefix in jobs:
                try:
                    generate_fractures(profile, dst_base, prefix,
                                       start=i, count=1, min_length=hard_profiles.get(profile, min_length))
                except DrawingError:
                    print('Exception generating for %s/%s' % (dst_base, prefix))
                    hard_profiles[profile] = hard_profiles.get(profile, min_length) / 2
                finally:
                    progress.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and simplify profile SVG files')
    parser.add_argument('source_dir', type=str,
                        help='Path to root directory for the profile SVG files (one '
                        'directory per profile type)')
    parser.add_argument('target_dir', type=str,
                        help='Path to directory for saving the fractures')
    parser.add_argument('count', type=int,
                        help='Number of fractures per profile')
    parser.add_argument('--min_length', type=float, default=20,
                        help='Smallest size (mm) of a fracture')
    parser.add_argument('--regular_y', action='store_true',
                        help='Is the top of the vessel at higher Y? (not the default)')
    args = parser.parse_args()

    run(src_dir=args.source_dir, dst_dir=args.target_dir,
        count=args.count, min_length=args.min_length,
        regular_y=args.regular_y)
