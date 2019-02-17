import json
import os
import pickle
import tempfile
import sys

from c3d.classification.dataset import FileId
from c3d.serving import webserver
from c3d.shape2.gen_pcl_dataset import SherdSVGDataset


class ShapeContext(object):
    def __init__(self, model_path, k=3, tmp_dir=None, regular_y=False):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.k = int(k)
        self.tmp_dir = os.path.abspath(tmp_dir)
        self.dataset = SherdSVGDataset(
            data_root=tmp_dir,  # Not used in practice
            regular_y=regular_y,
        )
        self.dataset.eval_mode = True
        self.dataset.balance = False
        self.dataset.data_spec = self.model.config.data_spec
        self.dataset.do_caching = False

    def classify_svg(self, blob):
        path = os.path.join(self.tmp_dir, 'input.svg')
        with open(path, 'wb') as f:
            f.write(blob)
        f_id = FileId(label='.', id='input.svg', augment=None, source_label=None)
        pts = self.dataset.prepare_or_cache(f_id)
        label_indices, label_scores = self.model.predict_top_k([pts], self.k, with_scores=True)
        label_indices, label_scores = list(label_indices[0]), [float(v) for v in label_scores[0]]
        return [(self.model.index_to_label[i], score) for i, score in zip(label_indices, label_scores)]


class ShapeClassificationServlet(webserver.ServerHandler):
    def server_init(self):
        self.register_server_handler(r'^/$', self.homepage)
        self.register_server_handler(r'^/classify_svg', self.classify_svg)

    def homepage(self, parameters):
        template_path = os.path.join(os.path.dirname(__file__), 'serving.html')
        with open(template_path, 'r') as f:
            return f.read()

    def classify_svg(self, parameters):
        name = parameters['name'].strip()
        data = parameters['data']
        scores = self.context.classify_svg(data)
        print('Scores: %s' % scores)
        return json.dumps({
            'name': name,
            'ranking': [{'class': pair[0], 'score': pair[1]}
                        for pair in scores],
        })


def main(argv):
    if len(argv) < 2 or len(argv) > 3 or '--help' in argv:
        print('Usage: %s model_path [k]' % os.path.basename(__file__))
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        context = ShapeContext(*sys.argv[1:], tmp_dir=tmpdir)
        webserver.serve_forever(8000, ShapeClassificationServlet, context)


if __name__ == '__main__':
    main(sys.argv)
