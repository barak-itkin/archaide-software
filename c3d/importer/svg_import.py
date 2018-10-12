import re
import c3d.datamodel

from xml.etree import ElementTree as ET


def parse_svg_path_d_tokens(d_string):
    # General information available at:
    #   https://www.w3.org/TR/SVG/paths.html#PathDataGeneralInformation
    # tl;dr:
    # - Letters for verbs followed by numbers
    # - Both space and comma can be used as separators everywhere
    for token in re.split(r'([a-z][^a-z]*)', d_string, flags=re.IGNORECASE):
        token = token.strip()
        if not token:
            continue
        verb = token[0]
        values = [float(v) for v in re.split(r'[\s,]+', token[1:]) if v]
        yield verb, values


def group_as_tuples(data, group_size):
    data = list(data)
    d = iter(data)
    while True:
        try:
            # PYTHON WARNING: DO NOT OMIT THE SQUARE BRACKETS!
            # If you will, it will send a generator expression whose yielding
            # of StopIteration will just make generation of empty tuples forever.
            # When we convert the expression to a list before that, it works
            # because the list comprehension is evaluated correctly.
            yield tuple([next(d) for i in range(group_size)])
        except StopIteration:
            break


def point_add(p1, p2):
    return tuple((v1 + v2) for v1, v2 in zip(p1, p2))


def make_relative_sequence(points, base):
    prev = base
    for p in points:
        current = point_add(prev, p)
        yield current
        prev = current


def parse_svg_path_d(d_string):
    """
    :return: An array of Outline objects, one for every path in the 'd' element
    """
    paths = []
    path = None
    last_point = (0, 0)
    # Loop over path verbs
    for verb, data in parse_svg_path_d_tokens(d_string):
        is_relative = verb.islower()
        # Move/Lineto verbs
        if verb in ('M', 'm', 'L', 'l'):
            points = group_as_tuples(data, 2)
            if is_relative:
                points = make_relative_sequence(points, last_point)
            if verb in ('M', 'm'):
                if path:
                    paths.append(c3d.datamodel.Outline(path))
                path = list(points)
            else:
                path.extend(points) # Add point tuples
            last_point = path[-1]
        # Vertical/Horizontal Lineto verbs
        elif verb in ('V', 'v', 'H', 'h'):
            is_vertical = verb in ('V', 'v')
            if is_relative:
                if is_vertical:
                    points = [(0, v) for v in data]
                else:
                    points = [(v, 0) for v in data]
                points = make_relative_sequence(points, last_point)
            else:
                if is_vertical:
                    points = [(last_point[0], v) for v in data]
                else:
                    points = [(v, last_point[1]) for v in data]
            path.extend(points) # Add point tuples
            last_point = path[-1]
        # Other verbs are not supported
        else:
            raise NotImplementedError('Unsupported SVG verb ' + verb)
    # Add the last path
    if path:
        paths.append(c3d.datamodel.Outline(path))
    return paths


def drawing_from_svg_document(document):
    """
    :param document: An ElementTree document/element, or a string representing the entire document
    :return:
    """
    if isinstance(document, str):
        document = ET.fromstring(document)

    if isinstance(document, ET.Element):
        root = document
    else:
        root = document.getroot()

    # The root may have a namespace, in which case the tag would be
    # '{http://www.w3.org/2000/svg}svg'
    if root.tag.startswith('{'): # Do we have a namespace?
        svg_namespace = root.tag.split('}')[0].lstrip('{') # Ugly
    else:
        svg_namespace = ''
    namespace_map = {'svg': svg_namespace}

    outlines = {}
    for path_element in root.findall('svg:g/svg:path', namespace_map):
        name = path_element.get('id')
        data = path_element.get('d')
        paths = parse_svg_path_d(data)
        if len(paths) != 1:
            raise ValueError('Unexpected SVG path element {} with {} distinct paths!'.format(name, len(paths)))
        outlines[name] = paths[0]

    properties = {}
    for text_element in root.findall('svg:g/svg:text', namespace_map):
        key, val = text_element.text.split(':', 1)
        properties[key.strip()] = val.strip()

    return c3d.datamodel.Drawing(outlines, properties)


@c3d.datamodel.with_read_handle(False)
def drawing_from_svg(f):
    return drawing_from_svg_document(ET.parse(f))
