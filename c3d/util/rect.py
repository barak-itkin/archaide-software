class CvRect(object):
    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height

    @property
    def xmax(self):
        return self.xmin + self.width - 1

    @property
    def ymax(self):
        return self.ymin + self.height - 1

    @staticmethod
    def img_bounds(img):
        return CvRect(xmin=0, ymin=0, width=img.shape[1], height=img.shape[0])

    def covers(self, other):
        return (
            self.xmin <= other.xmin and
            other.xmax <= self.xmax and
            self.ymin <= other.ymin and
            other.ymax <= self.ymax and
            True
        )

    def _as_tuple(self):
        return self.xmin, self.ymin, self.width, self.height

    def __hash__(self):
        return hash(self._as_tuple())

    def __eq__(self, other):
        if other is None or type(self) != type(other):
            return False
        return self._as_tuple() == other._as_tuple()

    def __repr__(self):
        return 'CvRect' + repr(self._as_tuple())

    def __str__(self):
        return '(%s, %s):[%s X %s]' % self._as_tuple()
