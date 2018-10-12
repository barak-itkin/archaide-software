import cgi
import http.server
import socketserver
import re
import traceback
import urllib.parse


class ServerHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.server_handlers = []
        self.server_init()
        super(ServerHandler, self).__init__(*args, **kwargs)

    def server_init(self):
        pass

    def register_server_handler(self, path_re, handler):
        self.server_handlers.append((re.compile(path_re), handler))

    def actual_do(self, path, parameters):
        for path_re, handler in self.server_handlers:
            if path_re.match(path):
                try:
                    result = handler(parameters)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(bytes(result, "utf8"))
                    return
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(bytes(traceback.format_exc(), "utf8"))
                    traceback.print_exc()
                    return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(bytes("Page not found :/", "utf8"))
        return

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        parameters = dict(urllib.parse.parse_qsl(parsed_url.query))
        return self.actual_do(path, parameters)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        parameters = {}
        if self.headers['content-type']:
            ctype, pdict = cgi.parse_header(self.headers['content-type'])
            if ctype == 'multipart/form-data':
                # https://stackoverflow.com/a/25091973
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'})
                parameters = dict((key, form[key].value) for key in form)
            elif ctype == 'application/x-www-form-urlencoded':
                length = int(self.headers['content-length'])
                qsl = str(self.rfile.read(length))
                for key, val in dict(urllib.parse.parse_qsl(qsl)).items():
                    parameters[key] = val
        return self.actual_do(path, parameters)


def serve_forever(port, handler_class, context_obj):
    class ServerContext(handler_class):
        @property
        def context(self):
            return context_obj

    httpd = socketserver.TCPServer(("", port), ServerContext)
    print("serving at port", port)
    httpd.serve_forever()