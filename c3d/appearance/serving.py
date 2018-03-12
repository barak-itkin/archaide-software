import json
import os

from c3d.serving import webserver


class AppearanceContext(object):
    def __init__(self):
        pass

    def classify_image(self, blob):
        return ['class1', 'class2', 'class3']


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


if __name__ == '__main__':
    context = AppearanceContext()
    webserver.serve_forever(8000, AppearanceClassificationServlet, context)
