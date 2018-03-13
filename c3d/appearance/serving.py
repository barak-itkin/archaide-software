import io
import json
import numpy as np
import os
import pickle
import PIL.Image
import sys

from . import make_dataset
from c3d.serving import webserver


class AppearanceContext(object):
    def __init__(self, model_path, k=3, resnet_dir=None):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        if resnet_dir:
            self.model.resnet_dir = resnet_dir
        self.model.load_resnet()
        self.k = int(k)

    def classify_image(self, blob):
        img = PIL.Image.open(io.BytesIO(blob)).convert('RGB')
        img = np.array(img)
        ready_img = make_dataset.prepare_img(img)
        print(ready_img.shape)
        print(ready_img.dtype)
        label_indices = self.model.predict_top_k([ready_img], self.k)[0]
        return [self.model.index_to_label[i] for i in label_indices]


class AppearanceClassificationServlet(webserver.ServerHandler):
    def server_init(self):
        self.register_server_handler(r'^/$', self.homepage)
        self.register_server_handler(r'^/classify_image', self.classify_image)

    def homepage(self, parameters):
        template_path = os.path.join(os.path.dirname(__file__), 'serving.html')
        with open(template_path, 'r') as f:
            return f.read()

    def classify_image(self, parameters):
        name = parameters['name'].strip()
        data = parameters['data']
        return json.dumps({
            name: name,
            'classes': self.context.classify_image(data)
        })


def main(argv):
    if len(argv) < 2 or len(argv) > 4 or '--help' in argv:
        print('Usage: %s model_path [k] [resnet_dir]')
        return
    context = AppearanceContext(*sys.argv[1:])
    webserver.serve_forever(8000, AppearanceClassificationServlet, context)


if __name__ == '__main__':
    main(sys.argv)