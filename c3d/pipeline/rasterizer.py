import argparse
import os
import traceback

import c3d.algorithm.fracture
import c3d.datamodel.profile
import c3d.graph.plot2d

from PIL import Image, ImageDraw


class FileHandler:
    def is_relevant(self, name):
        raise NotImplementedError()

    def get_outline(self, data):
        raise NotImplementedError()


class FractureHandler(FileHandler):
    def is_relevant(self, name):
        return name.endswith('.fracture.json')

    def get_outline(self, data):
        if isinstance(data, c3d.datamodel.Fracture):
            fracture = data
        elif isinstance(data, str):
            fracture = c3d.datamodel.loads_json(data, c3d.datamodel.Fracture)
        else:
            raise NotImplementedError()
        return c3d.algorithm.fracture.outline_from_fracture(fracture)


class ProfileHandler(FileHandler):
    def is_relevant(self, name):
        return name.endswith('.profile.json')

    def get_outline(self, data):
        if isinstance(data, c3d.datamodel.Profile):
            return data.outline
        elif isinstance(data, str):
            return c3d.datamodel.loads_json(data, c3d.datamodel.Profile).outline
        else:
            raise NotImplementedError()


class Profile2Handler(FileHandler):
    def is_relevant(self, name):
        return name.endswith('.fracture-prof2.json')

    def get_outline(self, data):
        if isinstance(data, c3d.datamodel.Profile2):
            return data.outline
        elif isinstance(data, str):
            prof = c3d.datamodel.loads_json(data, c3d.datamodel.Profile2) # type: c3d.datamodel.Profile2
            assert len(prof.outlines) == 1
            outline = prof.outlines[0]
            if outline.is_closed:
                valid = [
                    (outline.inside_map[i - 1] != c3d.datamodel.Outline2.NEITHER or
                     outline.inside_map[i - 0] != c3d.datamodel.Outline2.NEITHER)
                    for i in range(len(outline.inside_map))
                ]
            else:
                valid = [outline.inside_map[0] != c3d.datamodel.Outline2.NEITHER]
                valid += [
                    (outline.inside_map[i - 1] != c3d.datamodel.Outline2.NEITHER or
                     outline.inside_map[i - 0] != c3d.datamodel.Outline2.NEITHER)
                    for i in range(1, len(outline.inside_map))
                ]
                valid += [outline.inside_map[-1] != c3d.datamodel.Outline2.NEITHER]

            return c3d.datamodel.Outline([
                p for v, p in zip(valid, outline.points) if v
            ])
        else:
            raise NotImplementedError()


def get_bounding_box(outline):
    return c3d.graph.plot2d.Polygon(outline.points).compute_box()


class Rasterizer:
    def __init__(self, handler, width, height, antialias=True):
        self.handler = handler
        self.width_dst = width
        self.height_dst = height
        self.antialias = antialias

    # To anti-alias, we draw twice the size and then scale down
    @property
    def drawing_width(self):
        return self.width_dst if not self.antialias else self.width_dst * 2

    @property
    def drawing_height(self):
        return self.height_dst if not self.antialias else self.height_dst * 2

    def draw(self, data, drawing):
        outline = self.handler.get_outline(data)
        bbox = get_bounding_box(outline)

        # Compute the scale to fit the shape within the image
        # Take a 1px safety margin from each side to avoid aliasing
        # nonsense that will make the shape "spill-out"
        x_scale = (self.drawing_width - 2) / bbox.width
        y_scale = (self.drawing_height - 2) / bbox.height
        scale = min(x_scale, y_scale)

        # Center the drawing within the image bounds
        x_pad = (self.drawing_width - bbox.width * scale) / 2
        y_pad = (self.drawing_height - bbox.height * scale) / 2

        points = [
            ((x - bbox.min_x) * scale + x_pad,
             (y - bbox.min_y) * scale + y_pad)
            for x, y in outline.points
            ]

        drawing.polygon(points, fill=255, outline=None)

    def rasterize(self, data):
        img = Image.new('L', (self.drawing_width, self.drawing_height), color=0)
        drawing = ImageDraw.Draw(img)
        self.draw(data, drawing)

        if not self.antialias:
            return img
        else:
            small = img.resize((self.width_dst, self.height_dst), Image.ANTIALIAS)
            del img
            return small

    def batch_rasterize(self, basedir, force=False):
        # Create the base image outside of the loop, to avoid repeated memory
        # allocations when creating new images
        img = Image.new('L', (self.drawing_width, self.drawing_height), color=0)
        drawing = ImageDraw.Draw(img)

        for root, dirs, names in os.walk(basedir):
            for name in names:
                try:
                    if not self.handler.is_relevant(name):
                        continue

                    base_path = os.path.join(root, name)
                    out_path = os.path.join(root, '%s.png' % os.path.splitext(name)[0])

                    if not force and os.path.exists(out_path):
                        continue

                    # Clear our recycled image before drawing
                    drawing.rectangle(
                        [(0, 0), (self.drawing_width, self.drawing_height)],
                        fill=0, outline=None)

                    with open(base_path, 'r') as f:
                        self.draw(f.read(), drawing)

                    if not self.antialias:
                        img.save(out_path, 'PNG')
                    else:
                        small = img.resize((self.width_dst, self.height_dst), Image.ANTIALIAS)
                        small.save(out_path, 'PNG')
                        del small
                except:
                    print('Failed rasterizing %s' % name)
                    traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rasterize profile/fracture files')
    parser.add_argument('dir', type=str,
                        help='Path to directory containing the '
                        'profile/fracture files')
    parser.add_argument('type', type=str, choices=['profiles', 'fractures', 'new-fractures'],
                        help='The type of file to rasterize')
    parser.add_argument('--force', default=False, action='store_true',
                        help='Rasterize even if the image already exists')
    args = parser.parse_args()

    if args.type == 'profiles':
        rasterizer = Rasterizer(ProfileHandler(), 512, 512)
    elif args.type == 'fractures':
        rasterizer = Rasterizer(FractureHandler(), 256, 256)
    elif args.type == 'new-fractures':
        rasterizer = Rasterizer(Profile2Handler(), 256, 256)
    else:
        raise ValueError('Unknown type %s' % args.type)

    rasterizer.batch_rasterize(args.dir, force=args.force)
